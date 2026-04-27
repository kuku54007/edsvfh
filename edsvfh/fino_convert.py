from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from .checkpointing import atomic_write_json, load_json
from .progress import ETATracker
from .public_data import _generate_pick_place_episode


class FinoSourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class FailureShardManifest:
    source: str
    source_mode: str
    train_shards: list[str]
    calib_shards: list[str]
    eval_shards: list[str]
    train_episodes: int
    calib_episodes: int
    eval_episodes: int
    image_size: int
    episodes_per_shard: int
    manifest_path: str | None = None
    manifest_sha256: str | None = None
    resume_signature: str | None = None
    prefer_pseudo_onset: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            'source': self.source,
            'source_mode': self.source_mode,
            'train_shards': self.train_shards,
            'calib_shards': self.calib_shards,
            'eval_shards': self.eval_shards,
            'train_episodes': self.train_episodes,
            'calib_episodes': self.calib_episodes,
            'eval_episodes': self.eval_episodes,
            'image_size': self.image_size,
            'episodes_per_shard': self.episodes_per_shard,
            'manifest_path': self.manifest_path,
            'manifest_sha256': self.manifest_sha256,
            'resume_signature': self.resume_signature,
            'prefer_pseudo_onset': self.prefer_pseudo_onset,
        }


class _FailureShardSink:
    def __init__(self, split_dir: Path, prefix: str, episodes_per_shard: int, compression: str | None = 'gzip', *, resume_existing: bool = False) -> None:
        self.split_dir = split_dir
        self.prefix = prefix
        self.episodes_per_shard = episodes_per_shard
        self.compression = compression
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.shard_paths: list[str] = []
        self._current_file: h5py.File | None = None
        self._current_data_group: h5py.Group | None = None
        self._current_count = 0
        self._shard_index = 0
        self._episode_index = 0
        if resume_existing:
            self._resume_existing()

    def _resume_existing(self) -> None:
        shard_files = sorted(self.split_dir.glob(f'{self.prefix}_*.hdf5'))
        if not shard_files:
            return
        self.shard_paths = [str(p) for p in shard_files]
        counts = []
        for shard_path in shard_files:
            with h5py.File(shard_path, 'r') as f:
                counts.append(len(f['data'].keys()) if 'data' in f else 0)
        self._episode_index = int(sum(counts))
        self._shard_index = len(shard_files)
        if counts and counts[-1] < self.episodes_per_shard:
            last = shard_files[-1]
            self._current_file = h5py.File(last, 'a')
            self._current_data_group = self._current_file.require_group('data')
            self._current_count = int(counts[-1])
        else:
            self._current_count = 0

    @property
    def total_episodes(self) -> int:
        return int(self._episode_index)

    def _ensure_open(self) -> None:
        if self._current_file is not None and self._current_count < self.episodes_per_shard:
            return
        self.close()
        shard_path = self.split_dir / f'{self.prefix}_{self._shard_index:04d}.hdf5'
        self._current_file = h5py.File(shard_path, 'w')
        self._current_data_group = self._current_file.create_group('data')
        self._current_count = 0
        self._shard_index += 1
        self.shard_paths.append(str(shard_path))

    def add_episode(self, episode: dict[str, Any]) -> None:
        self._ensure_open()
        assert self._current_data_group is not None
        demo = self._current_data_group.create_group(f'demo_{self._episode_index:06d}')
        self._episode_index += 1
        self._current_count += 1

        demo.attrs['task'] = episode['task']
        demo.attrs['instruction'] = episode['instruction']
        demo.attrs['outcome'] = episode['outcome']
        demo.attrs['source_format'] = episode.get('source_format', 'fino_manifest')
        demo.attrs['episode_key'] = episode['episode_key']
        if episode.get('failure_onset', None) is not None:
            demo.attrs['failure_onset'] = int(episode['failure_onset'])
        if episode.get('original_failure_onset', None) is not None:
            demo.attrs['original_failure_onset'] = int(episode['original_failure_onset'])
        if episode.get('pseudo_failure_onset', None) is not None:
            demo.attrs['pseudo_failure_onset'] = int(episode['pseudo_failure_onset'])
        if episode.get('pseudo_onset_reason') not in (None, ''):
            demo.attrs['pseudo_onset_reason'] = str(episode['pseudo_onset_reason'])
        if episode.get('pseudo_onset_confidence', None) is not None:
            demo.attrs['pseudo_onset_confidence'] = float(episode['pseudo_onset_confidence'])

        compression = self.compression
        demo.create_dataset('actions', data=episode['actions'].astype(np.float32))
        demo.create_dataset('rewards', data=episode['rewards'].astype(np.float32))
        demo.create_dataset('dones', data=episode['dones'].astype(np.int32))
        obs = demo.create_group('obs')
        next_obs = demo.create_group('next_obs')

        obs.create_dataset('agentview_image', data=episode['images'].astype(np.uint8), compression=compression)
        obs.create_dataset('robot0_eef_pos', data=episode['eef'].astype(np.float32))
        obs.create_dataset('robot0_gripper_qpos', data=episode['gripper'].astype(np.float32))
        obs.create_dataset('object_pos', data=episode['object_pos'].astype(np.float32))
        obs.create_dataset('goal_pos', data=episode['goal_pos'].astype(np.float32))
        obs.create_dataset('policy_uncertainty', data=episode['uncertainty'].astype(np.float32))

        next_images = np.concatenate([episode['images'][1:], episode['images'][-1:]], axis=0)
        next_eef = np.concatenate([episode['eef'][1:], episode['eef'][-1:]], axis=0)
        next_gripper = np.concatenate([episode['gripper'][1:], episode['gripper'][-1:]], axis=0)
        next_obj = np.concatenate([episode['object_pos'][1:], episode['object_pos'][-1:]], axis=0)
        next_goal = np.concatenate([episode['goal_pos'][1:], episode['goal_pos'][-1:]], axis=0)
        next_unc = np.concatenate([episode['uncertainty'][1:], episode['uncertainty'][-1:]], axis=0)
        next_obs.create_dataset('agentview_image', data=next_images.astype(np.uint8), compression=compression)
        next_obs.create_dataset('robot0_eef_pos', data=next_eef.astype(np.float32))
        next_obs.create_dataset('robot0_gripper_qpos', data=next_gripper.astype(np.float32))
        next_obs.create_dataset('object_pos', data=next_obj.astype(np.float32))
        next_obs.create_dataset('goal_pos', data=next_goal.astype(np.float32))
        next_obs.create_dataset('policy_uncertainty', data=next_unc.astype(np.float32))

    def close(self) -> None:
        if self._current_file is not None:
            self._current_file.close()
            self._current_file = None
            self._current_data_group = None
            self._current_count = 0

def _hash_to_split(key: str, train_ratio: float = 0.7, calib_ratio: float = 0.15) -> str:
    digest = hashlib.sha1(key.encode('utf-8')).hexdigest()
    v = int(digest[:8], 16) / 0xFFFFFFFF
    if v < train_ratio:
        return 'train'
    if v < train_ratio + calib_ratio:
        return 'calib'
    return 'eval'


def _load_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == '.jsonl':
        rows = []
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and 'episodes' in data and isinstance(data['episodes'], list):
        return data['episodes']
    raise FinoSourceError('Manifest must be JSONL, a JSON list, or a JSON object containing an "episodes" list.')


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _expected_resume_signature(
    manifest_path: str | Path,
    *,
    source_label: str | None,
    source_mode: str,
    episodes_per_shard: int,
    image_size: int,
    prefer_pseudo_onset: bool,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest_sha256 = _sha256_file(manifest_path)
    payload = {
        'manifest_path': str(manifest_path),
        'manifest_sha256': manifest_sha256,
        'source': source_label or str(manifest_path),
        'source_mode': str(source_mode),
        'episodes_per_shard': int(episodes_per_shard),
        'image_size': int(image_size),
        'prefer_pseudo_onset': bool(prefer_pseudo_onset),
    }
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
    payload['resume_signature'] = signature
    return payload


def _resume_state_matches(state: dict[str, Any] | None, expected: dict[str, Any]) -> bool:
    if not state:
        return False
    signature = state.get('resume_signature')
    if signature is None:
        return False
    return str(signature) == str(expected['resume_signature'])


def _clear_existing_failure_conversion_state(output_dir: Path, manifest_out: Path, ckpt_path: Path) -> None:
    for split in ('train', 'calib', 'eval'):
        shutil.rmtree(output_dir / split, ignore_errors=True)
    for path in (manifest_out, ckpt_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _load_npy_or_default(path: str | None, default: np.ndarray, *, length: int, width: int) -> np.ndarray:
    if not path:
        return default
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] != length:
        raise FinoSourceError(f'Array length mismatch for {path}: expected {length}, got {arr.shape[0]}')
    if arr.shape[1] < width:
        arr = np.pad(arr, ((0, 0), (0, width - arr.shape[1])))
    return arr[:, :width].astype(np.float32)


def _read_images_from_entry(entry: dict[str, Any], image_size: int) -> np.ndarray:
    frame_paths: list[Path] = []
    if entry.get('frame_paths'):
        frame_paths = [Path(p) for p in entry['frame_paths']]
    else:
        frames_dir = entry.get('frames_dir')
        if not frames_dir:
            raise FinoSourceError('Each manifest entry must define either frame_paths or frames_dir.')
        frame_glob = entry.get('frame_glob', '*.png')
        frame_paths = sorted(Path(frames_dir).glob(frame_glob))
    if not frame_paths:
        raise FinoSourceError(f'No frames found for episode entry {entry.get("episode_id", "<unknown>")}')
    images = []
    for p in frame_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FinoSourceError(f'Failed to read image: {p}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (image_size, image_size):
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        images.append(img.astype(np.uint8))
    return np.asarray(images, dtype=np.uint8)


def _standardize_manifest_episode(entry: dict[str, Any], *, image_size: int, prefer_pseudo_onset: bool = False) -> dict[str, Any]:
    images = _read_images_from_entry(entry, image_size=image_size)
    length = int(images.shape[0])
    eef = _load_npy_or_default(entry.get('eef_npy'), np.zeros((length, 3), dtype=np.float32), length=length, width=3)
    gripper = _load_npy_or_default(entry.get('gripper_npy'), np.zeros((length, 1), dtype=np.float32), length=length, width=1)
    object_pos = _load_npy_or_default(entry.get('object_pos_npy'), np.zeros((length, 3), dtype=np.float32), length=length, width=3)
    goal_pos = _load_npy_or_default(entry.get('goal_pos_npy'), np.zeros((length, 3), dtype=np.float32), length=length, width=3)
    actions = _load_npy_or_default(entry.get('action_npy'), np.zeros((length, 4), dtype=np.float32), length=length, width=4)
    rewards = _load_npy_or_default(entry.get('reward_npy'), np.zeros((length, 1), dtype=np.float32), length=length, width=1).reshape(-1)
    uncertainty = _load_npy_or_default(entry.get('policy_uncertainty_npy'), np.zeros((length, 1), dtype=np.float32), length=length, width=1)
    dones = np.zeros((length,), dtype=np.int32)
    dones[-1] = 1

    original_failure_onset = entry.get('original_failure_onset', entry.get('failure_onset', None))
    original_failure_onset = None if original_failure_onset in (None, '', -1) else int(original_failure_onset)
    pseudo_failure_onset = entry.get('pseudo_failure_onset', None)
    pseudo_failure_onset = None if pseudo_failure_onset in (None, '', -1) else int(pseudo_failure_onset)
    failure_onset = pseudo_failure_onset if (prefer_pseudo_onset and pseudo_failure_onset is not None) else entry.get('failure_onset', None)
    failure_onset = None if failure_onset in (None, '', -1) else int(failure_onset)
    outcome = str(entry.get('outcome', 'failure' if failure_onset is not None else 'success'))
    episode_key = str(entry.get('episode_id', Path(entry.get('frames_dir', 'episode')).name))
    instruction = str(entry.get('instruction', 'Detect failures safely during the manipulation.'))
    task = str(entry.get('task', 'fino_failure'))

    return {
        'episode_key': episode_key,
        'task': task,
        'instruction': instruction,
        'outcome': outcome,
        'failure_onset': failure_onset,
        'original_failure_onset': original_failure_onset,
        'pseudo_failure_onset': pseudo_failure_onset,
        'pseudo_onset_reason': entry.get('pseudo_onset_reason'),
        'pseudo_onset_confidence': entry.get('pseudo_onset_confidence'),
        'images': images,
        'eef': eef,
        'gripper': gripper,
        'object_pos': object_pos,
        'goal_pos': goal_pos,
        'actions': actions,
        'rewards': rewards.astype(np.float32),
        'dones': dones,
        'uncertainty': uncertainty.astype(np.float32),
        'source_format': 'fino_manifest',
    }


def convert_failure_manifest_to_shards(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    source_label: str | None = None,
    source_mode: str = 'fino_manifest',
    episodes_per_shard: int = 32,
    image_size: int = 96,
    compression: str | None = 'gzip',
    show_progress: bool = True,
    checkpoint_path: str | Path | None = None,
    checkpoint_every_shards: int = 16,
    resume: bool = True,
    prefer_pseudo_onset: bool = False,
) -> FailureShardManifest:
    manifest_path = Path(manifest_path)
    entries = _load_manifest(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = output_dir / 'manifest.json'
    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else output_dir / '.convert_fino_state.json'
    expected_resume = _expected_resume_signature(
        manifest_path,
        source_label=source_label,
        source_mode=source_mode,
        episodes_per_shard=episodes_per_shard,
        image_size=image_size,
        prefer_pseudo_onset=prefer_pseudo_onset,
    )
    existing_manifest_state = load_json(manifest_out) if manifest_out.exists() else None
    existing_ckpt_state = load_json(ckpt_path) if ckpt_path.exists() else None

    if resume:
        manifest_matches = _resume_state_matches(existing_manifest_state, expected_resume)
        ckpt_matches = _resume_state_matches(existing_ckpt_state, expected_resume)
        stale_state_detected = (existing_manifest_state is not None and not manifest_matches) or (existing_ckpt_state is not None and not ckpt_matches)
        if stale_state_detected:
            print(
                f"[convert-fino] Detected stale conversion state for {manifest_path}; clearing {output_dir} before rerun.",
                flush=True,
            )
            _clear_existing_failure_conversion_state(output_dir, manifest_out, ckpt_path)
            existing_manifest_state = None
            existing_ckpt_state = None
        elif existing_manifest_state is not None and existing_ckpt_state is None and manifest_matches:
            return FailureShardManifest(**existing_manifest_state)
    sinks = {
        'train': _FailureShardSink(output_dir / 'train', 'failure_train', episodes_per_shard, compression=compression, resume_existing=resume),
        'calib': _FailureShardSink(output_dir / 'calib', 'failure_calib', episodes_per_shard, compression=compression, resume_existing=resume),
        'eval': _FailureShardSink(output_dir / 'eval', 'failure_eval', episodes_per_shard, compression=compression, resume_existing=resume),
    }
    counts = {split: sinks[split].total_episodes for split in ('train', 'calib', 'eval')}
    progress = ETATracker(
        label='convert-fino',
        total=len(entries),
        unit='episodes',
        print_every=max(1, episodes_per_shard // 2),
    ) if show_progress else None

    processed = int(sum(counts.values()))
    if progress is not None and processed > 0:
        progress.update(processed, extra=f"resume_from={processed} train={counts['train']} calib={counts['calib']} eval={counts['eval']}", force=True)
    for idx in range(processed, len(entries)):
        entry = entries[idx]
        episode = _standardize_manifest_episode(entry, image_size=image_size, prefer_pseudo_onset=prefer_pseudo_onset)
        split = entry.get('split')
        if split not in {'train', 'calib', 'eval'}:
            split = _hash_to_split(episode['episode_key'])
        sinks[split].add_episode(episode)
        counts[split] += 1
        processed += 1
        if progress is not None:
            progress.update(
                processed,
                extra=(
                    f"train={counts['train']} calib={counts['calib']} eval={counts['eval']} "
                    f"last_split={split} last_episode={episode['episode_key']}"
                ),
            )
        if checkpoint_every_shards > 0 and processed % checkpoint_every_shards == 0:
            atomic_write_json(ckpt_path, {
                'processed_total': processed,
                'counts': counts,
                'completed': False,
                'source': source_label or str(manifest_path),
                'source_mode': source_mode,
                'episodes_per_shard': episodes_per_shard,
                'image_size': image_size,
                'manifest_path': str(manifest_path),
                'manifest_sha256': expected_resume['manifest_sha256'],
                'prefer_pseudo_onset': bool(prefer_pseudo_onset),
                'resume_signature': expected_resume['resume_signature'],
            })

    for sink in sinks.values():
        sink.close()
    if progress is not None:
        progress.done(
            current=processed,
            extra=f"finished | train={counts['train']} calib={counts['calib']} eval={counts['eval']}",
        )

    manifest = FailureShardManifest(
        source=source_label or str(manifest_path),
        source_mode=source_mode,
        train_shards=sinks['train'].shard_paths,
        calib_shards=sinks['calib'].shard_paths,
        eval_shards=sinks['eval'].shard_paths,
        train_episodes=counts['train'],
        calib_episodes=counts['calib'],
        eval_episodes=counts['eval'],
        image_size=image_size,
        episodes_per_shard=episodes_per_shard,
        manifest_path=str(Path(manifest_path)),
        manifest_sha256=expected_resume['manifest_sha256'],
        resume_signature=expected_resume['resume_signature'],
        prefer_pseudo_onset=bool(prefer_pseudo_onset),
    )
    atomic_write_json(manifest_out, manifest.as_dict())
    atomic_write_json(ckpt_path, {
        'processed_total': processed,
        'counts': counts,
        'completed': True,
        'source': source_label or str(manifest_path),
        'source_mode': source_mode,
        'episodes_per_shard': episodes_per_shard,
        'image_size': image_size,
        'manifest_path': str(manifest_path),
        'manifest_sha256': expected_resume['manifest_sha256'],
        'prefer_pseudo_onset': bool(prefer_pseudo_onset),
        'resume_signature': expected_resume['resume_signature'],
    })
    return manifest


def create_mock_failure_manifest_dataset(
    root_dir: str | Path,
    output_dir: str | Path,
    *,
    num_episodes: int = 18,
    image_size: int = 96,
    episodes_per_shard: int = 4,
    seed: int = 13,
) -> FailureShardManifest:
    root_dir = Path(root_dir)
    episodes_root = root_dir / 'episodes'
    episodes_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    patterns = ['success', 'miss_grasp', 'drift', 'slip']
    entries: list[dict[str, Any]] = []
    for idx in range(num_episodes):
        kind = patterns[idx % len(patterns)]
        steps, failure_onset = _generate_pick_place_episode(kind, int(rng.integers(0, 10000)), length=24)
        ep_dir = episodes_root / f'ep_{idx:04d}_{kind}'
        frames_dir = ep_dir / 'rgb'
        frames_dir.mkdir(parents=True, exist_ok=True)
        length = len(steps)
        eef = np.zeros((length, 3), dtype=np.float32)
        gripper = np.zeros((length, 1), dtype=np.float32)
        object_pos = np.zeros((length, 3), dtype=np.float32)
        goal_pos = np.zeros((length, 3), dtype=np.float32)
        action = np.zeros((length, 4), dtype=np.float32)
        uncertainty = np.zeros((length, 1), dtype=np.float32)
        for t, step in enumerate(steps):
            obs = step.observation
            img = obs.image
            if img is None:
                img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            if img.shape[:2] != (image_size, image_size):
                img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(frames_dir / f'{t:06d}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            rs = np.asarray(obs.robot_state, dtype=np.float32).reshape(-1)
            if rs.size >= 3:
                eef[t] = rs[:3]
            if rs.size >= 4:
                gripper[t, 0] = rs[3]
            if rs.size >= 7:
                object_pos[t] = rs[4:7]
            if rs.size >= 10:
                goal_pos[t] = rs[7:10]
            act = np.asarray(obs.action, dtype=np.float32).reshape(-1)
            action[t, : min(4, act.size)] = act[:4]
            if obs.policy_stats.size >= 4:
                uncertainty[t, 0] = float(obs.policy_stats[3])
        np.save(ep_dir / 'eef.npy', eef)
        np.save(ep_dir / 'gripper.npy', gripper)
        np.save(ep_dir / 'object_pos.npy', object_pos)
        np.save(ep_dir / 'goal_pos.npy', goal_pos)
        np.save(ep_dir / 'action.npy', action)
        np.save(ep_dir / 'policy_uncertainty.npy', uncertainty)
        split = 'train' if idx < int(num_episodes * 0.7) else 'calib' if idx < int(num_episodes * 0.85) else 'eval'
        entries.append(
            {
                'episode_id': ep_dir.name,
                'split': split,
                'task': 'fino_failure_mock',
                'instruction': 'Detect whether the manipulation is heading toward failure.',
                'outcome': 'failure' if failure_onset is not None else 'success',
                'failure_onset': failure_onset,
                'frames_dir': str(frames_dir),
                'frame_glob': '*.png',
                'eef_npy': str(ep_dir / 'eef.npy'),
                'gripper_npy': str(ep_dir / 'gripper.npy'),
                'object_pos_npy': str(ep_dir / 'object_pos.npy'),
                'goal_pos_npy': str(ep_dir / 'goal_pos.npy'),
                'action_npy': str(ep_dir / 'action.npy'),
                'policy_uncertainty_npy': str(ep_dir / 'policy_uncertainty.npy'),
            }
        )
    manifest_path = root_dir / 'mock_fino_manifest.jsonl'
    with manifest_path.open('w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    return convert_failure_manifest_to_shards(
        manifest_path,
        output_dir,
        source_label='mock_fino_manifest',
        source_mode='mock_fino_manifest',
        episodes_per_shard=episodes_per_shard,
        image_size=image_size,
    )
