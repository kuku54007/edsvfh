from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .checkpointing import atomic_write_json, load_json
from .progress import ETATracker

FAILURE_HINTS = ("fail", "failure", "miss", "drift", "slip", "drop", "collision")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_OUTCOME_POS = {"success", "succ", "ok", "normal", "pass"}
_OUTCOME_NEG = {"failure", "fail", "miss", "drift", "slip", "drop", "collision", "abort", "error"}


def _infer_outcome_and_onset(ep_dir: Path) -> tuple[str, int | None]:
    metadata_candidates = [ep_dir / 'metadata.json', ep_dir / 'episode_meta.json', ep_dir / 'labels.json']
    for path in metadata_candidates:
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
            onset = data.get('failure_onset', data.get('onset', None))
            onset = None if onset in (None, '', -1) else int(onset)
            outcome = str(data.get('outcome', 'failure' if onset is not None else 'success'))
            return outcome, onset
    name = ep_dir.name.lower()
    if any(h in name for h in FAILURE_HINTS):
        return 'failure', None
    return 'success', None


def _discover_leaf_image_dirs(search_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for current, _dirs, files in os.walk(search_root):
        if any(Path(f).suffix.lower() in IMAGE_EXTS for f in files):
            candidates.append(Path(current))
    if not candidates:
        return []
    cset = set(candidates)
    leaves = [c for c in candidates if not any(other != c and c in other.parents for other in cset)]
    return sorted(set(leaves))


def _discover_episode_dirs(root_dir: Path) -> list[Path]:
    episodes_root = root_dir / 'episodes'
    if episodes_root.exists():
        candidates = [p for p in episodes_root.iterdir() if p.is_dir()]
        if candidates:
            return sorted(candidates)

    # Legacy project layout: episode/rgb/*.png
    candidates = []
    for rgb in root_dir.rglob('rgb'):
        if rgb.is_dir() and any(rgb.glob('*.png')):
            candidates.append(rgb.parent)
    if candidates:
        return sorted(set(candidates))

    # FINO / FAILURE layout: failnet_dataset/rgb_imgs/<task>/<episode-or-images>
    failnet_rgb_root = root_dir / 'failnet_dataset' / 'rgb_imgs'
    if failnet_rgb_root.exists():
        return _discover_leaf_image_dirs(failnet_rgb_root)

    # Generic fallback: any leaf directory containing image files.
    return _discover_leaf_image_dirs(root_dir)


def _choose_split(idx: int, total: int) -> str:
    train_cut = int(total * 0.7)
    calib_cut = int(total * 0.85)
    if idx < train_cut:
        return 'train'
    if idx < calib_cut:
        return 'calib'
    return 'eval'


def _default_checkpoint_path(output_path: Path) -> Path:
    return output_path.parent / f'{output_path.stem}.manifest_ckpt.json'


def _normalize_task_name(name: str) -> str:
    return name.strip().lower().replace('-', '_').replace(' ', '_')


def _load_failnet_annotations(root_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Best-effort parser for FINO FAILURE annotation txt files.

    The public dataset ships task-specific files such as `place_annotation.txt`.
    The exact row schema may vary, so this parser intentionally stays tolerant:
    it extracts identifier-like tokens from each non-empty row and marks them as
    failure examples unless the row explicitly contains a success keyword.
    """
    annotations: dict[tuple[str, str], dict[str, Any]] = {}
    for txt in sorted(root_dir.glob('*_annotation.txt')):
        task = _normalize_task_name(txt.stem.replace('_annotation', ''))
        try:
            lines = txt.read_text(encoding='utf-8').splitlines()
        except UnicodeDecodeError:
            lines = txt.read_text(encoding='latin-1').splitlines()

        for raw in lines:
            raw = raw.strip()
            if not raw or raw.startswith('#'):
                continue
            tokens = [tok for tok in re.split(r'[\s,;]+', raw) if tok]
            if not tokens:
                continue

            low_tokens = [tok.lower() for tok in tokens]
            if any(tok in _OUTCOME_NEG for tok in low_tokens):
                outcome = 'failure'
            elif any(tok in _OUTCOME_POS for tok in low_tokens):
                outcome = 'success'
            else:
                # In the released annotation files, rows usually enumerate failure ids.
                outcome = 'failure'

            onset = None
            onset_match = re.search(r'(?:onset|start|failure(?:_time)?)\s*[=:]?\s*(\d+)', raw, flags=re.IGNORECASE)
            if onset_match:
                onset = int(onset_match.group(1))

            keys: set[str] = set()
            for tok in tokens:
                clean = tok.strip().strip('"\'')
                if not clean:
                    continue
                low = clean.lower()
                if low in _OUTCOME_POS or low in _OUTCOME_NEG:
                    continue
                keys.add(clean)
                keys.add(clean.lower())
                keys.add(Path(clean).stem)
                keys.add(Path(clean).name)
                if clean.isdigit():
                    num = int(clean)
                    for width in (2, 3, 4, 5, 6):
                        keys.add(f"{num:0{width}d}")

            if not keys:
                keys.add(raw)

            for key in keys:
                annotations[(task, key)] = {'outcome': outcome, 'failure_onset': onset}
    return annotations


def _match_failnet_annotation(ep_dir: Path, annotations: dict[tuple[str, str], dict[str, Any]], root_dir: Path) -> tuple[str, int | None] | None:
    failnet_rgb_root = root_dir / 'failnet_dataset' / 'rgb_imgs'
    if not failnet_rgb_root.exists():
        return None
    try:
        rel = ep_dir.relative_to(failnet_rgb_root)
    except Exception:
        return None
    if not rel.parts:
        return None

    task = _normalize_task_name(rel.parts[0])
    candidate_keys: set[str] = set()
    candidate_keys.add(ep_dir.name)
    candidate_keys.add(ep_dir.name.lower())
    candidate_keys.add(rel.as_posix())
    candidate_keys.add(rel.as_posix().lower())
    if len(rel.parts) > 1:
        candidate_keys.add('/'.join(rel.parts[1:]))
        candidate_keys.add('/'.join(rel.parts[1:]).lower())
        candidate_keys.add(rel.parts[-1])
        candidate_keys.add(rel.parts[-1].lower())
    num_match = re.search(r'(\d+)', ep_dir.name)
    if num_match:
        num = int(num_match.group(1))
        for width in (1, 2, 3, 4, 5, 6):
            candidate_keys.add(f"{num:0{width}d}")

    for key in candidate_keys:
        ann = annotations.get((task, key))
        if ann is not None:
            return ann['outcome'], ann.get('failure_onset')
    return None


def generate_fino_manifest_from_episode_dirs(
    root_dir: str | Path,
    output_path: str | Path,
    *,
    frame_glob: str = '*.png',
    task: str = 'fino_failure',
    instruction: str = 'Detect whether the manipulation is heading toward failure.',
    show_progress: bool = True,
    checkpoint_path: str | Path | None = None,
    checkpoint_every_shards: int = 32,
    resume: bool = True,
) -> Path:
    root_dir = Path(root_dir)
    output_path = Path(output_path)
    episodes = _discover_episode_dirs(root_dir)
    if not episodes:
        raise FileNotFoundError(f'No episode directories found under {root_dir}')

    failnet_annotations = _load_failnet_annotations(root_dir)
    failnet_rgb_root = root_dir / 'failnet_dataset' / 'rgb_imgs'

    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else _default_checkpoint_path(output_path)
    rows: list[dict[str, Any]] = []
    start_idx = 0
    if resume and ckpt_path.exists() and output_path.exists():
        state = load_json(ckpt_path)
        start_idx = int(state.get('next_index', 0))
        with output_path.open('r', encoding='utf-8') as f:
            rows = [json.loads(line) for line in f if line.strip()]

    total = len(episodes)
    progress = ETATracker(label='generate-fino-manifest', total=total, unit='episodes', print_every=max(1, total // 10), initial_current=0) if show_progress else None
    if progress is not None and start_idx > 0:
        progress.update(start_idx, extra=f'resume_from={start_idx}', force=True)

    for idx in range(start_idx, len(episodes)):
        ep_dir = episodes[idx]
        if (ep_dir / 'rgb').exists() and any((ep_dir / 'rgb').glob(frame_glob)):
            frames_dir = ep_dir / 'rgb'
        elif any(ep_dir.glob(frame_glob)):
            frames_dir = ep_dir
        else:
            continue

        matched = _match_failnet_annotation(ep_dir, failnet_annotations, root_dir)
        if matched is not None:
            outcome, failure_onset = matched
        else:
            outcome, failure_onset = _infer_outcome_and_onset(ep_dir)

        episode_task = task
        if failnet_rgb_root.exists():
            try:
                rel = ep_dir.relative_to(failnet_rgb_root)
                if rel.parts:
                    episode_task = _normalize_task_name(rel.parts[0])
            except Exception:
                pass

        rows.append({
            'episode_id': ep_dir.name,
            'split': _choose_split(idx, total),
            'task': episode_task,
            'instruction': instruction,
            'outcome': outcome,
            'failure_onset': failure_onset,
            'frames_dir': str(frames_dir),
            'frame_glob': frame_glob,
            'eef_npy': str(ep_dir / 'eef.npy') if (ep_dir / 'eef.npy').exists() else None,
            'gripper_npy': str(ep_dir / 'gripper.npy') if (ep_dir / 'gripper.npy').exists() else None,
            'object_pos_npy': str(ep_dir / 'object_pos.npy') if (ep_dir / 'object_pos.npy').exists() else None,
            'goal_pos_npy': str(ep_dir / 'goal_pos.npy') if (ep_dir / 'goal_pos.npy').exists() else None,
            'action_npy': str(ep_dir / 'action.npy') if (ep_dir / 'action.npy').exists() else None,
            'policy_uncertainty_npy': str(ep_dir / 'policy_uncertainty.npy') if (ep_dir / 'policy_uncertainty.npy').exists() else None,
        })
        if progress is not None:
            progress.update(idx + 1, extra=f"episode={ep_dir.name}")
        if checkpoint_every_shards > 0 and ((idx + 1) % checkpoint_every_shards == 0):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open('w', encoding='utf-8') as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            atomic_write_json(ckpt_path, {'next_index': idx + 1, 'completed': False, 'root_dir': str(root_dir)})

    if progress is not None:
        progress.done(current=len(rows), extra='manifest scan finished')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    atomic_write_json(ckpt_path, {'next_index': len(episodes), 'completed': True, 'root_dir': str(root_dir)})
    return output_path
