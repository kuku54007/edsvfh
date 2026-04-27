from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import h5py
import numpy as np

from .checkpointing import atomic_write_json, load_json
from .config import AppConfig
from .fino_convert import convert_failure_manifest_to_shards
from .fino_finetune import fine_tune_bundle_on_failure_shards
from .fiper_pseudo_onset import (
    PseudoOnsetRebuildResult,
    fit_droid_success_baseline,
    relabel_fino_manifest_with_pseudo_onsets,
)
from .progress import ETATracker
from .droid_convert import (
    DroidPreparedTFDSSource,
    _infer_outcome_from_metadata as _infer_rlds_outcome_from_metadata,
    _maybe_resize as _rlds_maybe_resize,
    _pick_image as _rlds_pick_image,
    _pick_instruction as _rlds_pick_instruction,
)

_FAILURE_PATH_MARKERS = {'failure', 'failures', 'failed', 'not_success', 'not_successful', 'unsuccessful'}
_SUCCESS_PATH_MARKERS = {'success', 'successful'}
_DEFAULT_CAMERA_PREFERENCE = ('exterior_image_1_left', 'exterior_image_2_left', 'wrist_image_left', 'ext1', 'ext2', 'wrist')


class DroidFailureSourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class DroidFailureManifestResult:
    manifest_path: str
    source_root: str
    frames_root: str
    total_episodes: int
    failure_episodes: int
    skipped_episodes: int
    max_episodes: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            'manifest_path': self.manifest_path,
            'source_root': self.source_root,
            'frames_root': self.frames_root,
            'total_episodes': int(self.total_episodes),
            'failure_episodes': int(self.failure_episodes),
            'skipped_episodes': int(self.skipped_episodes),
            'max_episodes': self.max_episodes,
        }


def infer_droid_raw_outcome_from_path(path: str | Path) -> str:
    parts = {str(p).lower() for p in Path(path).parts}
    if parts & _FAILURE_PATH_MARKERS:
        return 'failure'
    if parts & _SUCCESS_PATH_MARKERS:
        return 'success'
    return 'unknown'


def _metadata_scalar_to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _metadata_scalar_to_python(value.item())
        if value.size == 1:
            return _metadata_scalar_to_python(value.reshape(-1)[0])
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore')
    return value


def _metadata_bool(value: Any) -> bool | None:
    value = _metadata_scalar_to_python(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
    text = str(value).strip().lower()
    if text in {'1', 'true', 'yes', 'y', 'success', 'successful'}:
        return True
    if text in {'0', 'false', 'no', 'n', 'failure', 'failed', 'not_success', 'not_successful', 'unsuccessful'}:
        return False
    return None


def infer_droid_raw_outcome(metadata: dict[str, Any] | None = None, path: str | Path | None = None, *, default: str = 'unknown') -> str:
    metadata = metadata or {}
    for key in ('outcome', 'result', 'label', 'episode_outcome', 'trajectory_outcome'):
        if key not in metadata:
            continue
        value = _metadata_scalar_to_python(metadata.get(key))
        text = str(value).strip().lower()
        if text in {'failure', 'failed', 'fail', 'not_success', 'not_successful', 'unsuccessful', 'false', '0'}:
            return 'failure'
        if text in {'success', 'successful', 'true', '1'}:
            return 'success'
    for key in ('success', 'is_success', 'successful', 'task_success', 'episode_success'):
        if key not in metadata:
            continue
        flag = _metadata_bool(metadata.get(key))
        if flag is not None:
            return 'success' if flag else 'failure'
    for key in ('file_path', 'recording_folderpath', 'episode_path', 'path'):
        value = _metadata_scalar_to_python(metadata.get(key))
        if value is None:
            continue
        inferred = infer_droid_raw_outcome_from_path(str(value))
        if inferred != 'unknown':
            return inferred
    if path is not None:
        inferred = infer_droid_raw_outcome_from_path(path)
        if inferred != 'unknown':
            return inferred
    return default


def discover_droid_raw_failure_episodes(root_dir: str | Path, *, max_episodes: int | None = None) -> list[Path]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f'DROID raw root not found: {root}')
    episodes: list[Path] = []
    for h5_path in sorted(root.rglob('trajectory.h5')):
        ep_dir = h5_path.parent
        metadata, _ = _load_metadata(ep_dir)
        if infer_droid_raw_outcome(metadata, ep_dir) != 'failure':
            continue
        episodes.append(ep_dir)
        if max_episodes is not None and len(episodes) >= int(max_episodes):
            break
    return episodes


def _safe_stem(text: str, *, fallback: str) -> str:
    text = str(text).strip() or fallback
    text = re.sub(r'[^A-Za-z0-9_.+-]+', '_', text).strip('._')
    return text or fallback


def _load_metadata(episode_dir: Path) -> tuple[dict[str, Any], Path | None]:
    for path in sorted(episode_dir.glob('metadata_*.json')):
        try:
            return json.loads(path.read_text(encoding='utf-8')), path
        except Exception:
            continue
    return {}, None


def _episode_id_from_metadata_or_path(episode_dir: Path, metadata_path: Path | None, root_dir: Path) -> str:
    if metadata_path is not None:
        stem = metadata_path.stem
        return _safe_stem(stem[len('metadata_'):] if stem.startswith('metadata_') else stem, fallback='droid_failure')
    try:
        rel = str(episode_dir.relative_to(root_dir))
    except ValueError:
        rel = str(episode_dir)
    digest = hashlib.sha1(rel.encode('utf-8')).hexdigest()[:12]
    return _safe_stem(f'{episode_dir.name}_{digest}', fallback=f'droid_failure_{digest}')


def _pick_instruction(metadata: dict[str, Any]) -> str:
    for key in ('language_instruction', 'language_instruction_1', 'language_instruction_2', 'language_instruction_3', 'instruction', 'task_instruction', 'task', 'task_name', 'prompt'):
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = next((v for v in value if str(v).strip()), '')
        if isinstance(value, bytes):
            value = value.decode('utf-8', errors='ignore')
        value = str(value).strip()
        if value:
            return value
    return 'Monitor the DROID manipulation trajectory and predict whether it is entering a failure state.'


def _pick_task(metadata: dict[str, Any]) -> str:
    for key in ('task', 'task_name', 'task_type', 'primitive', 'skill'):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return _safe_stem(str(value), fallback='droid_not_successful')
    return 'droid_not_successful'


def _camera_rank(path: Path, preference: Sequence[str]) -> tuple[int, int, str]:
    name = path.name.lower()
    stereo_penalty = 1 if 'stereo' in name else 0
    for idx, key in enumerate(preference):
        if str(key).lower() in name:
            return idx, stereo_penalty, name
    return len(preference), stereo_penalty, name


def _choose_mp4(episode_dir: Path, camera_preference: Sequence[str]) -> Path:
    mp4_root = episode_dir / 'recordings' / 'MP4'
    candidates = sorted(mp4_root.rglob('*.mp4')) if mp4_root.exists() else []
    if not candidates:
        candidates = sorted(episode_dir.rglob('*.mp4'))
    if not candidates:
        raise DroidFailureSourceError(f'No MP4 video found for DROID raw episode: {episode_dir}')
    return sorted(candidates, key=lambda p: _camera_rank(p, camera_preference))[0]


def _extract_video_frames(video_path: Path, frames_dir: Path, *, image_size: int, frame_stride: int, max_frames: int | None, overwrite: bool) -> list[Path]:
    existing = sorted(frames_dir.glob('*.png'))
    if existing and not overwrite:
        return existing[: max_frames if max_frames is not None else None]
    frames_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for p in existing:
            p.unlink(missing_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise DroidFailureSourceError(f'Failed to open video: {video_path}')
    out: list[Path] = []
    raw_idx = 0
    saved_idx = 0
    stride = max(1, int(frame_stride))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if raw_idx % stride == 0:
                frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
                out_path = frames_dir / f'{saved_idx:06d}.png'
                cv2.imwrite(str(out_path), frame)
                out.append(out_path)
                saved_idx += 1
                if max_frames is not None and saved_idx >= int(max_frames):
                    break
            raw_idx += 1
    finally:
        cap.release()
    if not out:
        raise DroidFailureSourceError(f'No frames were extracted from video: {video_path}')
    return out


def _dataset_to_array(dataset: h5py.Dataset) -> np.ndarray:
    arr = np.asarray(dataset[()])
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr.astype(np.float32)


def _find_h5_dataset(h5: h5py.File, candidates: Sequence[str], suffixes: Sequence[str]) -> np.ndarray | None:
    for name in candidates:
        if name in h5 and isinstance(h5[name], h5py.Dataset):
            return _dataset_to_array(h5[name])
    found: list[tuple[int, str, h5py.Dataset]] = []
    def visitor(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        low = name.lower()
        for rank, suffix in enumerate(suffixes):
            if low.endswith(suffix.lower()):
                found.append((rank, name, obj))
                return
    h5.visititems(visitor)
    if not found:
        return None
    _, _, ds = sorted(found, key=lambda item: (item[0], item[1]))[0]
    return _dataset_to_array(ds)


def _resample_time_major(arr: np.ndarray | None, *, length: int, width: int, default: float = 0.0) -> np.ndarray:
    if arr is None or np.asarray(arr).size == 0:
        return np.full((length, width), default, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[0] == length:
        out = arr
    else:
        idx = np.linspace(0, arr.shape[0] - 1, num=length)
        idx = np.clip(np.rint(idx).astype(np.int64), 0, arr.shape[0] - 1)
        out = arr[idx]
    if out.shape[1] < width:
        out = np.pad(out, ((0, 0), (0, width - out.shape[1])), constant_values=default)
    return out[:, :width].astype(np.float32)


def _load_low_dim_arrays(trajectory_h5: Path, *, length: int) -> dict[str, np.ndarray]:
    with h5py.File(trajectory_h5, 'r') as h5:
        eef_raw = _find_h5_dataset(h5, ('observation/cartesian_position', 'observations/cartesian_position', 'obs/cartesian_position', 'action/cartesian_position', 'cartesian_position'), ('/observation/cartesian_position', '/observations/cartesian_position', '/cartesian_position'))
        grip_raw = _find_h5_dataset(h5, ('observation/gripper_position', 'observations/gripper_position', 'obs/gripper_position', 'action/gripper_position', 'gripper_position'), ('/observation/gripper_position', '/observations/gripper_position', '/gripper_position'))
        joint_raw = _find_h5_dataset(h5, ('observation/joint_position', 'observations/joint_position', 'obs/joint_position', 'action/joint_position', 'joint_position'), ('/observation/joint_position', '/observations/joint_position', '/joint_position'))
        action_raw = _find_h5_dataset(h5, ('action/cartesian_position', 'action/joint_velocity', 'action/joint_position', 'actions', 'action'), ('/action/cartesian_position', '/action/joint_velocity', '/action/joint_position', '/actions', '/action'))
    eef = _resample_time_major(eef_raw, length=length, width=3)
    gripper = _resample_time_major(grip_raw, length=length, width=1)
    joint = _resample_time_major(joint_raw, length=length, width=7)
    width = max(4, int(action_raw.shape[1]) if action_raw is not None and action_raw.ndim >= 2 else 4)
    action = _resample_time_major(action_raw, length=length, width=width)
    return {'eef': eef, 'gripper': gripper, 'joint': joint, 'action': action}


def _motion_uncertainty_proxy(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    out = np.zeros((action.shape[0],), dtype=np.float32)
    if action.ndim == 2 and action.shape[0] > 1 and action.shape[1] > 0:
        signal = action[:, : min(action.shape[1], 6)]
        delta = np.linalg.norm(np.diff(signal, axis=0), axis=1).astype(np.float32)
        scale = float(np.quantile(delta, 0.90)) if delta.size else 0.0
        if scale <= 1e-6:
            scale = float(np.max(delta)) if delta.size else 0.0
        out[1:] = np.clip(delta / scale, 0.0, 1.0) if scale > 1e-6 else 0.0
    return out.reshape(-1, 1).astype(np.float32)


def _write_arrays(ep_out: Path, arrays: dict[str, np.ndarray]) -> dict[str, str]:
    ep_out.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for key, arr in arrays.items():
        path = ep_out / f'{key}.npy'
        np.save(path, np.asarray(arr, dtype=np.float32))
        out[key] = str(path)
    return out


def _hash_to_split(key: str, train_ratio: float = 0.70, calib_ratio: float = 0.15) -> str:
    value = int(hashlib.sha1(key.encode('utf-8')).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < train_ratio:
        return 'train'
    if value < train_ratio + calib_ratio:
        return 'calib'
    return 'eval'


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _manifest_partial_path(path: Path) -> Path:
    return path.with_name(path.name + '.partial')


def _write_manifest_scan_checkpoint(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    atomic_write_json(path, payload)


def _manifest_scan_state(
    checkpoint_path: Path | None,
    partial_manifest_path: Path,
    output_path: Path,
    *,
    source_root: str | Path,
    frames_root: str | Path,
    scanned: int,
    skipped: int,
    rows: Sequence[dict[str, Any]],
    max_episodes: int | None,
    scan_max_episodes: int | None,
    finished: bool,
) -> dict[str, Any]:
    return {
        "source_root": str(source_root),
        "frames_root": str(frames_root),
        "output": str(output_path),
        "partial_output": str(partial_manifest_path),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "scanned": int(scanned),
        "kept": int(len(rows)),
        "skipped": int(skipped),
        "max_episodes": max_episodes,
        "scan_max_episodes": scan_max_episodes,
        "finished": bool(finished),
    }


def _checkpoint_manifest_scan(
    checkpoint_path: Path | None,
    partial_manifest_path: Path,
    output_path: Path,
    *,
    source_root: str | Path,
    frames_root: str | Path,
    scanned: int,
    skipped: int,
    rows: Sequence[dict[str, Any]],
    max_episodes: int | None,
    scan_max_episodes: int | None,
    finished: bool,
) -> None:
    if checkpoint_path is None:
        return
    _write_jsonl(partial_manifest_path, rows)
    _write_manifest_scan_checkpoint(
        checkpoint_path,
        _manifest_scan_state(
            checkpoint_path,
            partial_manifest_path,
            output_path,
            source_root=source_root,
            frames_root=frames_root,
            scanned=scanned,
            skipped=skipped,
            rows=rows,
            max_episodes=max_episodes,
            scan_max_episodes=scan_max_episodes,
            finished=finished,
        ),
    )


def _resume_manifest_scan(
    checkpoint_path: Path | None,
    partial_manifest_path: Path,
    output_path: Path,
    *,
    resume: bool,
) -> tuple[list[dict[str, Any]], int, int]:
    if not resume or checkpoint_path is None or not checkpoint_path.exists():
        return [], 0, 0
    try:
        payload = load_json(checkpoint_path)
    except Exception:
        return [], 0, 0
    if bool(payload.get("finished", False)) and output_path.exists():
        return _read_jsonl(output_path), int(payload.get("scanned", 0)), int(payload.get("skipped", 0))
    source_path = partial_manifest_path if partial_manifest_path.exists() else (output_path if output_path.exists() else None)
    rows = _read_jsonl(source_path) if source_path is not None else []
    return rows, int(payload.get("scanned", 0)), int(payload.get("skipped", 0))



def _episode_key_from_rlds_metadata(metadata: dict[str, Any], raw_episode: dict[str, Any]) -> str:
    for key in ('file_path', 'recording_folderpath', 'episode_path', 'path'):
        value = _metadata_scalar_to_python(metadata.get(key))
        if value is not None and str(value).strip():
            return str(value)
    return f"rlds_episode_{int(raw_episode.get('episode_index', 0))}"


def _episode_id_from_rlds_metadata(metadata: dict[str, Any], raw_episode: dict[str, Any]) -> str:
    key = _episode_key_from_rlds_metadata(metadata, raw_episode)
    digest = hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]
    tail = Path(key).name or Path(str(_metadata_scalar_to_python(metadata.get('recording_folderpath', '')))).name
    tail = _safe_stem(tail, fallback='droid_rlds_failure')
    return _safe_stem(f'{tail}_{digest}', fallback=f'droid_rlds_failure_{digest}')


def _scalar_float(value: Any, default: float = 0.0) -> float:
    value = _metadata_scalar_to_python(value)
    if value is None:
        return float(default)
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size:
            return float(arr[0])
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return float(default)


def _array_from_obs(obs: dict[str, Any], key: str, *, width: int, default: float = 0.0) -> np.ndarray:
    arr = np.asarray(obs.get(key, np.full((width,), default, dtype=np.float32)), dtype=np.float32).reshape(-1)
    if arr.size < width:
        arr = np.pad(arr, (0, width - arr.size), constant_values=default)
    return arr[:width].astype(np.float32)


def _action_from_step(step: dict[str, Any]) -> np.ndarray:
    if 'action' in step:
        arr = np.asarray(step.get('action'), dtype=np.float32).reshape(-1)
    else:
        action_dict = step.get('action_dict', {}) or {}
        for key in ('cartesian_position', 'joint_velocity', 'joint_position'):
            if key in action_dict:
                arr = np.asarray(action_dict[key], dtype=np.float32).reshape(-1)
                break
        else:
            arr = np.zeros((4,), dtype=np.float32)
    if arr.size < 4:
        arr = np.pad(arr, (0, 4 - arr.size))
    return arr.astype(np.float32)


def _write_rlds_episode_assets(
    raw_episode: dict[str, Any],
    ep_out: Path,
    *,
    image_size: int,
    frame_stride: int,
    max_frames: int | None,
    camera_preference: Sequence[str],
    overwrite_frames: bool,
) -> tuple[Path, dict[str, str], int]:
    step_list = raw_episode.get('steps', [])
    if not isinstance(step_list, list):
        raise DroidFailureSourceError('Expected RLDS episode steps to be decoded as a list.')
    frames_dir = ep_out / 'rgb'
    existing = sorted(frames_dir.glob('*.png'))
    if overwrite_frames and existing:
        for path in existing:
            path.unlink(missing_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    eef: list[np.ndarray] = []
    gripper: list[np.ndarray] = []
    joint: list[np.ndarray] = []
    action: list[np.ndarray] = []
    reward: list[np.ndarray] = []
    explicit_uncertainty: list[np.ndarray] = []
    saved_count = 0
    stride = max(1, int(frame_stride))

    for raw_idx, step in enumerate(step_list):
        if raw_idx % stride != 0:
            continue
        obs = step.get('observation', {}) or {}
        image = _rlds_maybe_resize(_rlds_pick_image(obs, tuple(camera_preference)), image_size)
        if image is None:
            continue
        if max_frames is not None and saved_count >= int(max_frames):
            break

        out_path = frames_dir / f'{saved_count:06d}.png'
        if not out_path.exists() or overwrite_frames:
            cv2.imwrite(str(out_path), cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        eef.append(_array_from_obs(obs, 'cartesian_position', width=3))
        gripper.append(_array_from_obs(obs, 'gripper_position', width=1))
        joint.append(_array_from_obs(obs, 'joint_position', width=7))
        action.append(_action_from_step(step))
        reward.append(np.array([_scalar_float(step.get('reward', 0.0))], dtype=np.float32))
        explicit_uncertainty.append(np.array([_scalar_float(step.get('policy_uncertainty', 0.0))], dtype=np.float32))
        saved_count += 1

    if saved_count <= 0:
        raise DroidFailureSourceError('RLDS failure episode did not contain usable RGB frames.')
    arrays = {
        'eef': np.stack(eef, axis=0).astype(np.float32),
        'gripper': np.stack(gripper, axis=0).astype(np.float32),
        'joint': np.stack(joint, axis=0).astype(np.float32),
        'action': np.stack(action, axis=0).astype(np.float32),
        'reward': np.stack(reward, axis=0).astype(np.float32),
    }
    explicit_unc = np.stack(explicit_uncertainty, axis=0).astype(np.float32)
    if np.max(np.abs(explicit_unc)) > 1e-8:
        arrays['policy_uncertainty'] = explicit_unc
    else:
        arrays['policy_uncertainty'] = _motion_uncertainty_proxy(arrays['action'])
    npy_paths = _write_arrays(ep_out, arrays)
    return frames_dir, npy_paths, saved_count


def generate_droid_failure_manifest_from_episode_source(
    source: Any,
    output: str | Path,
    *,
    source_root: str | Path,
    frames_root: str | Path | None = None,
    image_size: int = 96,
    frame_stride: int = 2,
    max_episodes: int | None = None,
    max_frames_per_episode: int | None = None,
    camera_preference: Sequence[str] = _DEFAULT_CAMERA_PREFERENCE,
    overwrite_frames: bool = False,
    show_progress: bool = True,
    scan_max_episodes: int | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 256,
    resume: bool = True,
) -> DroidFailureManifestResult:
    """Generate a FINO-style failure manifest from prepared DROID RLDS episodes.

    For long-running RLDS scans, this function can periodically checkpoint a
    partial manifest plus scan cursor so disconnected terminals do not force a
    full restart.
    """
    output = Path(output)
    frames_root_path = Path(frames_root) if frames_root is not None else output.parent / 'droid_failure_frames'
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    partial_output = _manifest_partial_path(output)
    total_estimate = None
    try:
        total_estimate = source.estimate_num_episodes()
    except Exception:
        total_estimate = None
    progress_total = min(total_estimate, scan_max_episodes) if (total_estimate is not None and scan_max_episodes is not None) else total_estimate
    progress = ETATracker(label='generate-droid-rlds-failure-manifest', total=progress_total, unit='episodes', print_every=25) if show_progress else None

    rows, scanned, skipped = _resume_manifest_scan(checkpoint, partial_output, output, resume=resume)
    if show_progress and scanned > 0:
        print(f'[generate-droid-rlds-failure-manifest] resume scanned={scanned} kept={len(rows)} skipped={skipped}', flush=True)
    if max_episodes is not None and len(rows) >= int(max_episodes):
        _write_jsonl(output, rows[: int(max_episodes)])
        if partial_output.exists():
            partial_output.unlink()
        _write_manifest_scan_checkpoint(
            checkpoint,
            _manifest_scan_state(
                checkpoint,
                partial_output,
                output,
                source_root=source_root,
                frames_root=frames_root_path,
                scanned=scanned,
                skipped=skipped,
                rows=rows[: int(max_episodes)],
                max_episodes=max_episodes,
                scan_max_episodes=scan_max_episodes,
                finished=True,
            ),
        )
        return DroidFailureManifestResult(str(output), str(source_root), str(frames_root_path), scanned, int(max_episodes), skipped, max_episodes)

    iterator = source.iter_episodes(start_episode=scanned)
    for raw_episode in iterator:
        if scan_max_episodes is not None and scanned >= int(scan_max_episodes):
            break
        scanned += 1
        metadata = raw_episode.get('episode_metadata', {}) or {}
        outcome = _infer_rlds_outcome_from_metadata(metadata, default='unknown')
        if outcome != 'failure':
            if progress is not None:
                progress.update(scanned, extra=f'kept={len(rows)} skipped={skipped} last=non_failure')
            if checkpoint is not None and checkpoint_every > 0 and scanned % int(checkpoint_every) == 0:
                _checkpoint_manifest_scan(
                    checkpoint,
                    partial_output,
                    output,
                    source_root=source_root,
                    frames_root=frames_root_path,
                    scanned=scanned,
                    skipped=skipped,
                    rows=rows,
                    max_episodes=max_episodes,
                    scan_max_episodes=scan_max_episodes,
                    finished=False,
                )
            continue
        episode_id = _episode_id_from_rlds_metadata(metadata, raw_episode)
        ep_out = frames_root_path / episode_id
        try:
            frames_dir, npy_paths, n_frames = _write_rlds_episode_assets(
                raw_episode,
                ep_out,
                image_size=image_size,
                frame_stride=frame_stride,
                max_frames=max_frames_per_episode,
                camera_preference=camera_preference,
                overwrite_frames=overwrite_frames,
            )
            key = _episode_key_from_rlds_metadata(metadata, raw_episode)
            rows.append({
                'episode_id': episode_id,
                'split': _hash_to_split(episode_id),
                'task': _pick_task(metadata),
                'instruction': _rlds_pick_instruction(raw_episode.get('steps', [])) or _pick_instruction(metadata),
                'outcome': 'failure',
                'failure_onset': None,
                'frames_dir': str(frames_dir),
                'frame_glob': '*.png',
                'eef_npy': npy_paths['eef'],
                'gripper_npy': npy_paths['gripper'],
                'action_npy': npy_paths['action'],
                'reward_npy': npy_paths.get('reward'),
                'policy_uncertainty_npy': npy_paths['policy_uncertainty'],
                'source_dataset': 'DROID',
                'source_format': 'droid_rlds_not_successful',
                'droid_rlds_episode_index': int(raw_episode.get('episode_index', -1)),
                'droid_episode_key': key,
                'droid_file_path': str(_metadata_scalar_to_python(metadata.get('file_path'))) if metadata.get('file_path') is not None else None,
                'droid_recording_folderpath': str(_metadata_scalar_to_python(metadata.get('recording_folderpath'))) if metadata.get('recording_folderpath') is not None else None,
                'failure_label_source': 'droid_rlds_metadata_path',
                'num_frames': int(n_frames),
            })
        except Exception as exc:
            skipped += 1
            if show_progress:
                key = _episode_key_from_rlds_metadata(metadata, raw_episode)
                print(f'[generate-droid-rlds-failure-manifest] skipped {key}: {exc}', flush=True)
        if progress is not None:
            progress.update(scanned, extra=f'kept={len(rows)} skipped={skipped}')
        if checkpoint is not None and checkpoint_every > 0 and scanned % int(checkpoint_every) == 0:
            _checkpoint_manifest_scan(
                checkpoint,
                partial_output,
                output,
                source_root=source_root,
                frames_root=frames_root_path,
                scanned=scanned,
                skipped=skipped,
                rows=rows,
                max_episodes=max_episodes,
                scan_max_episodes=scan_max_episodes,
                finished=False,
            )
        if max_episodes is not None and len(rows) >= int(max_episodes):
            break
    if progress is not None:
        progress.done(current=scanned, extra=f'finished | kept={len(rows)} skipped={skipped}')
    _write_jsonl(output, rows)
    if partial_output.exists():
        partial_output.unlink()
    _write_manifest_scan_checkpoint(
        checkpoint,
        _manifest_scan_state(
            checkpoint,
            partial_output,
            output,
            source_root=source_root,
            frames_root=frames_root_path,
            scanned=scanned,
            skipped=skipped,
            rows=rows,
            max_episodes=max_episodes,
            scan_max_episodes=scan_max_episodes,
            finished=True,
        ),
    )
    return DroidFailureManifestResult(str(output), str(source_root), str(frames_root_path), scanned, len(rows), skipped, max_episodes)


def generate_droid_failure_manifest_from_rlds(
    source: str | Path,
    output: str | Path,
    *,
    split: str = 'train',
    dataset_name: str | None = None,
    version: str | None = None,
    frames_root: str | Path | None = None,
    image_size: int = 96,
    frame_stride: int = 2,
    max_episodes: int | None = None,
    scan_max_episodes: int | None = None,
    max_frames_per_episode: int | None = None,
    camera_preference: Sequence[str] = _DEFAULT_CAMERA_PREFERENCE,
    overwrite_frames: bool = False,
    show_progress: bool = True,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 256,
    resume: bool = True,
) -> DroidFailureManifestResult:
    episode_source = DroidPreparedTFDSSource(
        source=source,
        split=split,
        dataset_name=dataset_name,
        version=version,
        max_episodes=scan_max_episodes,
    )
    return generate_droid_failure_manifest_from_episode_source(
        episode_source,
        output,
        source_root=source,
        frames_root=frames_root,
        image_size=image_size,
        frame_stride=frame_stride,
        max_episodes=max_episodes,
        max_frames_per_episode=max_frames_per_episode,
        camera_preference=camera_preference,
        overwrite_frames=overwrite_frames,
        show_progress=show_progress,
        scan_max_episodes=scan_max_episodes,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        resume=resume,
    )

def generate_droid_failure_manifest_from_raw(root_dir: str | Path, output: str | Path, *, frames_root: str | Path | None = None, image_size: int = 96, frame_stride: int = 2, max_episodes: int | None = None, max_frames_per_episode: int | None = None, camera_preference: Sequence[str] = _DEFAULT_CAMERA_PREFERENCE, overwrite_frames: bool = False, show_progress: bool = True) -> DroidFailureManifestResult:
    root = Path(root_dir)
    output = Path(output)
    frames_root_path = Path(frames_root) if frames_root is not None else output.parent / 'droid_failure_frames'
    episode_dirs = discover_droid_raw_failure_episodes(root, max_episodes=max_episodes)
    rows: list[dict[str, Any]] = []
    skipped = 0
    progress = ETATracker(label='generate-droid-failure-manifest', total=len(episode_dirs), unit='episodes', print_every=1) if show_progress else None
    for idx, ep_dir in enumerate(episode_dirs):
        metadata, metadata_path = _load_metadata(ep_dir)
        episode_id = _episode_id_from_metadata_or_path(ep_dir, metadata_path, root)
        ep_out = frames_root_path / episode_id
        try:
            video_path = _choose_mp4(ep_dir, camera_preference)
            frames_dir = ep_out / 'rgb'
            frame_paths = _extract_video_frames(video_path, frames_dir, image_size=image_size, frame_stride=frame_stride, max_frames=max_frames_per_episode, overwrite=overwrite_frames)
            arrays = _load_low_dim_arrays(ep_dir / 'trajectory.h5', length=len(frame_paths))
            arrays['policy_uncertainty'] = _motion_uncertainty_proxy(arrays['action'])
            npy_paths = _write_arrays(ep_out, arrays)
            path_outcome = infer_droid_raw_outcome_from_path(ep_dir)
            label_source = 'droid_raw_failure_folder' if path_outcome == 'failure' else 'droid_raw_metadata'
            rows.append({
                'episode_id': episode_id,
                'split': _hash_to_split(episode_id),
                'task': _pick_task(metadata),
                'instruction': _pick_instruction(metadata),
                'outcome': 'failure',
                'failure_onset': None,
                'frames_dir': str(frames_dir),
                'frame_glob': '*.png',
                'eef_npy': npy_paths['eef'],
                'gripper_npy': npy_paths['gripper'],
                'action_npy': npy_paths['action'],
                'policy_uncertainty_npy': npy_paths['policy_uncertainty'],
                'source_dataset': 'DROID',
                'source_format': 'droid_raw_not_successful',
                'droid_episode_dir': str(ep_dir),
                'droid_trajectory_h5': str(ep_dir / 'trajectory.h5'),
                'droid_video_mp4': str(video_path),
                'droid_metadata_json': str(metadata_path) if metadata_path is not None else None,
                'failure_label_source': label_source,
            })
        except Exception as exc:
            skipped += 1
            if show_progress:
                print(f'[generate-droid-failure-manifest] skipped {ep_dir}: {exc}', flush=True)
        if progress is not None:
            progress.update(idx + 1, extra=f'kept={len(rows)} skipped={skipped}')
    if progress is not None:
        progress.done(current=len(episode_dirs), extra=f'finished | kept={len(rows)} skipped={skipped}')
    _write_jsonl(output, rows)
    return DroidFailureManifestResult(str(output), str(root), str(frames_root_path), len(episode_dirs), len(rows), skipped, max_episodes)


def rebuild_droid_failure_with_pseudo_onset(droid_success_shard_dir: str | Path, droid_failure_manifest_path: str | Path, base_bundle: str | Path, *, baseline_output_path: str | Path, pseudo_manifest_output_path: str | Path, converted_output_dir: str | Path, output_bundle_path: str | Path, config: AppConfig | None = None, epochs: int = 3, feature_source: str = 'visual', window: int = 3, phase_bins: int = 10, quantile: float = 0.97, min_phase_count: int = 8, image_size: int = 96, update_scaler: bool = False, show_progress: bool = True, fit_max_episodes: int | None = None, replace_failure_onset: bool = True, prefer_pseudo_onset: bool = True, episodes_per_shard: int = 32, pseudo_checkpoint_path: str | Path | None = None, pseudo_checkpoint_every: int = 32, pseudo_resume: bool = True, droid_failure_checkpoint_path: str | Path | None = None, droid_failure_checkpoint_every_shards: int = 1, droid_failure_resume: bool = True) -> PseudoOnsetRebuildResult:
    config = config or AppConfig()
    fit_droid_success_baseline(droid_success_shard_dir, output_path=baseline_output_path, config=config, feature_source=feature_source, window=window, phase_bins=phase_bins, quantile=quantile, min_phase_count=min_phase_count, max_episodes=fit_max_episodes, show_progress=show_progress)
    relabel_fino_manifest_with_pseudo_onsets(droid_failure_manifest_path, baseline_output_path, pseudo_manifest_output_path, image_size=image_size, replace_failure_onset=replace_failure_onset, config=config, show_progress=show_progress, checkpoint_path=pseudo_checkpoint_path, checkpoint_every=pseudo_checkpoint_every, resume=pseudo_resume)
    convert_failure_manifest_to_shards(pseudo_manifest_output_path, converted_output_dir, source_label=str(pseudo_manifest_output_path), source_mode='droid_not_successful_pseudo_onset', episodes_per_shard=episodes_per_shard, image_size=image_size, show_progress=show_progress, prefer_pseudo_onset=prefer_pseudo_onset)
    result = fine_tune_bundle_on_failure_shards(base_bundle, converted_output_dir, output_path=output_bundle_path, config=config, epochs=epochs, update_scaler=update_scaler, show_progress=show_progress, checkpoint_path=droid_failure_checkpoint_path, checkpoint_every_shards=droid_failure_checkpoint_every_shards, resume=droid_failure_resume)
    return PseudoOnsetRebuildResult(str(Path(baseline_output_path)), str(Path(pseudo_manifest_output_path)), str(Path(converted_output_dir)), str(Path(output_bundle_path)), result.metrics, result.strategies)
