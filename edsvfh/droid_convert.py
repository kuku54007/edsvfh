from __future__ import annotations

import hashlib
import json
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Protocol

from .progress import ETATracker
from .checkpointing import atomic_write_json, load_json
from .config import AppConfig
from .encoders import build_encoder
from .public_data import infer_action_type
from .types import StepObservation

import cv2
import h5py
import numpy as np


DEFAULT_CAMERA_PREFERENCE = (
    'exterior_image_1_left',
    'exterior_image_2_left',
    'wrist_image_left',
)


class DroidSourceError(RuntimeError):
    pass


class OptionalTFDSDependencyMissingError(DroidSourceError):
    pass


class DroidEpisodeSource(Protocol):
    def iter_episodes(self, start_episode: int = 0) -> Iterable[dict]: ...
    def estimate_num_episodes(self) -> int | None: ...


@dataclass(frozen=True)
class DroidShardManifest:
    source: str
    source_mode: str
    train_shards: list[str]
    calib_shards: list[str]
    eval_shards: list[str]
    train_episodes: int
    calib_episodes: int
    eval_episodes: int
    image_size: int
    step_stride: int
    episodes_per_shard: int
    action_space: str

    def as_dict(self) -> dict:
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
            'step_stride': self.step_stride,
            'episodes_per_shard': self.episodes_per_shard,
            'action_space': self.action_space,
        }


@dataclass(frozen=True)
class DroidPreparedTFDSSource:
    source: str | Path
    split: str = 'train'
    dataset_name: str | None = None
    version: str | None = None
    max_episodes: int | None = None

    @staticmethod
    def _ensure_protobuf_descriptor_compat() -> None:
        """Backfill deprecated protobuf FieldDescriptor.label for TFDS readers.

        Newer protobuf Python runtimes removed ``FieldDescriptor.label`` in favor
        of ``is_repeated`` / ``is_required``. TFDS read-only builders still reach
        for ``field.label`` when parsing prepared ``features.json`` metadata.
        This shim makes protobuf >=7 interoperate long enough for TFDS builder
        reconstruction to succeed.
        """
        try:
            from google.protobuf import descriptor as pb_descriptor
        except Exception:
            return
        if hasattr(pb_descriptor.FieldDescriptor, 'label'):
            return

        def _compat_label(self):
            if getattr(self, 'is_repeated', False):
                return pb_descriptor.FieldDescriptor.LABEL_REPEATED
            if getattr(self, 'is_required', False):
                return pb_descriptor.FieldDescriptor.LABEL_REQUIRED
            return pb_descriptor.FieldDescriptor.LABEL_OPTIONAL

        try:
            setattr(pb_descriptor.FieldDescriptor, 'label', property(_compat_label))
        except Exception:
            # If the runtime prevents monkey-patching, callers should pin protobuf<7.
            pass

    def _import_tfds(self):
        self._ensure_protobuf_descriptor_compat()
        try:
            import os
            if os.getenv('EDSVFH_TF_DISABLE_GPU', '0') == '1':
                try:
                    import tensorflow as tf  # type: ignore
                    try:
                        tf.config.set_visible_devices([], 'GPU')
                    except Exception:
                        pass
                except Exception:
                    pass
            import tensorflow_datasets as tfds
        except Exception as exc:  # pragma: no cover - optional dependency
            raise OptionalTFDSDependencyMissingError(
                'tensorflow-datasets is required to read prepared DROID RLDS builders. Install the tfds extra.'
            ) from exc
        return tfds

    @staticmethod
    def _discover_builder_dirs_local(root: Path) -> list[Path]:
        if (root / 'features.json').exists():
            return [root]
        matches = sorted({p.parent for p in root.rglob('features.json')})
        return matches

    def _load_builder(self):  # pragma: no cover - optional dependency
        tfds = self._import_tfds()
        source = str(self.source)
        # If dataset_name is explicitly supplied, we let TFDS reconstruct via the builder name.
        if self.dataset_name:
            kwargs = {'data_dir': source}
            if self.version:
                kwargs['version'] = self.version
            return tfds.builder(self.dataset_name, **kwargs)

        if source.startswith('gs://'):
            # We accept the GCS path as-is. Whether the path works depends on the
            # caller's TensorFlow/GCS authentication environment.
            try:
                return tfds.builder_from_directory(source)
            except Exception:
                raise DroidSourceError(
                    'For gs:// sources, pass a concrete TFDS builder directory or use --dataset-name droid if your environment registers the dataset builder.'
                )

        root = Path(source)
        if not root.exists():
            raise FileNotFoundError(f'DROID source path not found: {root}')
        builder_dirs = self._discover_builder_dirs_local(root)
        if not builder_dirs:
            raise DroidSourceError(
                'Could not find a prepared TFDS builder directory under the given path. Expected a folder containing features.json.'
            )
        if len(builder_dirs) == 1:
            return tfds.builder_from_directory(builder_dirs[0])
        return tfds.builder_from_directories(builder_dirs)

    @staticmethod
    def _to_numpy_leaf(value):
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (str, bytes, bytearray)):
            return value
        if isinstance(value, (np.generic, int, float, bool)):
            return value
        if hasattr(value, 'numpy'):
            try:
                return value.numpy()
            except Exception:
                pass
        try:
            return np.asarray(value)
        except Exception:
            return value

    @classmethod
    def _normalize_nested(cls, value):
        if isinstance(value, dict):
            return {k: cls._normalize_nested(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            try:
                arr = np.asarray(value)
                if arr.dtype != object:
                    return arr
            except Exception:
                pass
            return [cls._normalize_nested(v) for v in value]
        return cls._to_numpy_leaf(value)

    @classmethod
    def _steps_to_list(cls, steps, tfds) -> list[dict]:  # pragma: no cover - optional dependency
        # RLDS may expose steps as a nested tf.data.Dataset, as a dict of arrays,
        # or (after an outer tfds.as_numpy call) as a tfds _IterableDataset.
        if isinstance(steps, dict):
            keys = list(steps.keys())
            length = len(next(iter(steps.values()))) if keys else 0
            out = []
            for i in range(length):
                step = {}
                for key, value in steps.items():
                    if isinstance(value, dict):
                        step[key] = {k: cls._to_numpy_leaf(v[i]) for k, v in value.items()}
                    else:
                        step[key] = cls._to_numpy_leaf(value[i])
                out.append(step)
            return out

        # Nested TFDS iterables may already yield numpy-like dicts after the outer
        # episode dataset has been wrapped with tfds.as_numpy(...). In that case
        # calling tfds.as_numpy again raises a TypeError on _IterableDataset.
        if hasattr(steps, '__iter__') and not isinstance(steps, (str, bytes, bytearray, np.ndarray)):
            out = []
            for step in steps:
                out.append(cls._normalize_nested(step))
            return out

        return [cls._normalize_nested(step) for step in tfds.as_numpy(steps)]



    def estimate_num_episodes(self) -> int | None:  # pragma: no cover - optional dependency
        try:
            builder = self._load_builder()
            split_info = builder.info.splits.get(self.split)
            if split_info is None:
                return self.max_episodes
            total = int(split_info.num_examples)
            return min(total, self.max_episodes) if self.max_episodes is not None else total
        except Exception:
            return self.max_episodes

    def iter_episodes(self, start_episode: int = 0) -> Iterator[dict]:  # pragma: no cover - optional dependency
        tfds = self._import_tfds()
        builder = self._load_builder()
        ds = builder.as_dataset(split=self.split)
        if start_episode > 0:
            ds = ds.skip(int(start_episode))
        if self.max_episodes is not None:
            remaining = max(0, int(self.max_episodes) - int(start_episode))
            ds = ds.take(remaining)
        iterable = tfds.as_numpy(ds)
        for offset, episode in enumerate(iterable):
            ep_idx = int(start_episode) + offset
            steps = self._steps_to_list(episode['steps'], tfds)
            metadata = episode.get('episode_metadata', {})
            yield {
                'episode_metadata': metadata,
                'steps': steps,
                'episode_index': ep_idx,
            }


@dataclass(frozen=True)
class MockDroidEpisodeSource:
    num_episodes: int = 18
    steps_per_episode: int = 20
    image_size: int = 96
    include_failures: bool = True
    seed: int = 5



    def estimate_num_episodes(self) -> int | None:
        return int(self.num_episodes)

    def iter_episodes(self, start_episode: int = 0) -> Iterator[dict]:
        rng = np.random.default_rng(self.seed)
        for ep_idx in range(start_episode, self.num_episodes):
            instruction = 'pick up the object and move it to the target area'
            outcome = 'failure' if self.include_failures and ep_idx % 5 == 0 else 'success'
            failure_onset = None
            steps: list[dict] = []
            eef = np.array([0.20, 0.25, 0.10], dtype=np.float32)
            obj = np.array([0.40, 0.45, 0.10], dtype=np.float32)
            goal = np.array([0.72, 0.72, 0.12], dtype=np.float32)
            gripper = 0.9
            for t in range(self.steps_per_episode):
                frac = t / max(self.steps_per_episode - 1, 1)
                if t < 5:
                    eef = eef + 0.35 * (obj - eef) * 0.35
                elif t < 8:
                    gripper = 0.15
                    eef = obj.copy()
                elif t < 12:
                    eef[2] = min(0.36, eef[2] + 0.06)
                    obj = eef.copy()
                elif t < 17:
                    eef = eef + 0.25 * (goal - eef)
                    obj = eef.copy()
                else:
                    gripper = 0.9
                    obj = goal.copy()
                    eef = goal.copy()
                if outcome == 'failure' and t >= 11:
                    if failure_onset is None:
                        failure_onset = 11
                    obj[2] = max(0.10, obj[2] - 0.05)
                    eef[2] = max(0.12, eef[2] - 0.04)
                action = np.concatenate([
                    (eef + rng.normal(0, 0.005, size=3)).astype(np.float32),
                    np.zeros(3, dtype=np.float32),
                    np.array([gripper], dtype=np.float32),
                ])
                image = _render_droid_like_frame(eef, obj, goal, gripper, size=self.image_size)
                step = {
                    'language_instruction': instruction,
                    'reward': np.float32(1.0 if t == self.steps_per_episode - 1 and outcome == 'success' else 0.0),
                    'action': action,
                    'observation': {
                        'gripper_position': np.array([gripper], dtype=np.float32),
                        'cartesian_position': np.concatenate([eef, np.zeros(3, dtype=np.float32)]).astype(np.float32),
                        'joint_position': np.linspace(0, 0.5 + frac, 7, dtype=np.float32),
                        'exterior_image_1_left': image,
                        'exterior_image_2_left': image[:, ::-1].copy(),
                        'wrist_image_left': image,
                    },
                    'is_last': np.bool_(t == self.steps_per_episode - 1),
                    'is_terminal': np.bool_(t == self.steps_per_episode - 1),
                }
                if outcome == 'failure':
                    unc = min(1.0, 0.2 + 0.1 * max(0, t - 9))
                else:
                    unc = 0.05 + 0.02 * (t in (5, 7, 12))
                step['policy_uncertainty'] = np.float32(unc)
                steps.append(step)
            yield {
                'episode_metadata': {
                    'file_path': f'mock://episode_{ep_idx}_{outcome}',
                    'recording_folderpath': f'mock://episode_{ep_idx}',
                    'outcome': outcome,
                    'failure_onset': failure_onset,
                },
                'steps': steps,
                'episode_index': ep_idx,
            }


def _render_droid_like_frame(eef: np.ndarray, obj: np.ndarray, goal: np.ndarray, gripper: float, size: int = 96) -> np.ndarray:
    img = np.full((size, size, 3), 235, dtype=np.uint8)

    def to_px(p: np.ndarray) -> tuple[int, int]:
        x = int(np.clip(p[0], 0.0, 1.0) * (size - 1))
        y = int((1.0 - np.clip(p[1], 0.0, 1.0)) * (size - 1))
        return x, y

    cv2.circle(img, to_px(goal), 10, (0, 180, 0), -1)
    cv2.circle(img, to_px(obj), 8, (0, 0, 220), -1)
    color = (200, 60, 0) if gripper < 0.45 else (60, 60, 60)
    cv2.circle(img, to_px(eef), 7, color, -1)
    cv2.line(img, to_px(eef), to_px(obj), (180, 180, 180), 1)
    return img


def _pick_instruction(step_list: list[dict]) -> str:
    for key in ('language_instruction', 'language_instruction_2', 'language_instruction_3'):
        for step in step_list:
            value = step.get(key)
            if value is not None:
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='ignore')
                value = str(value).strip()
                if value:
                    return value
    return 'Complete the manipulation task safely.'


def _pick_image(observation: dict, camera_preference: tuple[str, ...]) -> np.ndarray | None:
    for key in camera_preference:
        if key in observation:
            image = np.asarray(observation[key])
            if image.ndim == 3:
                return image
    return None


def _maybe_resize(image: np.ndarray | None, image_size: int) -> np.ndarray | None:
    if image is None:
        return None
    if image.shape[0] == image_size and image.shape[1] == image_size:
        return image.astype(np.uint8)
    return cv2.resize(image.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_AREA)


def _hash_to_split(key: str, train_ratio: float = 0.8, calib_ratio: float = 0.1) -> str:
    digest = hashlib.sha1(key.encode('utf-8')).hexdigest()
    v = int(digest[:8], 16) / 0xFFFFFFFF
    if v < train_ratio:
        return 'train'
    if v < train_ratio + calib_ratio:
        return 'calib'
    return 'eval'



def _metadata_scalar_to_python(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore')
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.shape == ():
            return _metadata_scalar_to_python(value.item())
        if value.size == 1:
            return _metadata_scalar_to_python(value.reshape(-1)[0])
        return [_metadata_scalar_to_python(v) for v in value.reshape(-1).tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _metadata_bool(value) -> bool | None:
    value = _metadata_scalar_to_python(value)
    if value is None:
        return None
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


def _infer_outcome_from_metadata(metadata: dict, *, default: str = 'success') -> str:
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
    path_markers = {'failure', 'failures', 'failed', 'not_success', 'not_successful', 'unsuccessful'}
    success_markers = {'success', 'successful'}
    for key in ('file_path', 'recording_folderpath', 'episode_path', 'path'):
        value = _metadata_scalar_to_python(metadata.get(key))
        if value is None:
            continue
        parts = {part.lower() for part in Path(str(value)).parts}
        if parts & path_markers:
            return 'failure'
        if parts & success_markers:
            return 'success'
    return default


def _infer_failure_onset_from_metadata(metadata: dict) -> int | None:
    for key in ('pseudo_failure_onset', 'failure_onset', 'failure_start', 'onset', 'failure_frame', 'failure_step'):
        if key not in metadata:
            continue
        value = _metadata_scalar_to_python(metadata.get(key))
        if value in (None, '', -1):
            continue
        try:
            onset = int(float(value))
        except (TypeError, ValueError):
            continue
        if onset >= 0:
            return onset
    return None

def _standardize_droid_episode(
    raw_episode: dict,
    *,
    image_size: int,
    step_stride: int,
    action_space: str,
    camera_preference: tuple[str, ...],
) -> dict:
    step_list = raw_episode['steps']
    if not isinstance(step_list, list):
        raise TypeError('Expected raw_episode["steps"] to be a list of step dicts after source decoding.')
    instruction = _pick_instruction(step_list)
    metadata = raw_episode.get('episode_metadata', {})
    file_path = metadata.get('file_path', f'episode_{raw_episode.get("episode_index", 0)}')
    if isinstance(file_path, bytes):
        file_path = file_path.decode('utf-8', errors='ignore')
    episode_key = str(file_path)
    outcome = _infer_outcome_from_metadata(metadata, default='success')
    failure_onset = _infer_failure_onset_from_metadata(metadata)

    images = []
    eef = []
    gripper = []
    joint = []
    actions = []
    rewards = []
    dones = []
    uncertainty = []
    for idx, step in enumerate(step_list[:: max(1, step_stride)]):
        obs = step.get('observation', {})
        image = _maybe_resize(_pick_image(obs, camera_preference), image_size)
        cart = np.asarray(obs.get('cartesian_position', np.zeros(6, dtype=np.float32)), dtype=np.float32).reshape(-1)
        if cart.size < 3:
            cart = np.pad(cart, (0, 3 - cart.size))
        eef.append(cart[:3].astype(np.float32))
        grip = np.asarray(obs.get('gripper_position', np.zeros(1, dtype=np.float32)), dtype=np.float32).reshape(-1)
        gripper.append(np.array([float(grip[0]) if grip.size else 0.0], dtype=np.float32))
        joint_pos = np.asarray(obs.get('joint_position', np.zeros(7, dtype=np.float32)), dtype=np.float32).reshape(-1)
        if joint_pos.size < 7:
            joint_pos = np.pad(joint_pos, (0, 7 - joint_pos.size))
        joint.append(joint_pos[:7].astype(np.float32))
        if action_space == 'joint_position':
            act = np.concatenate([joint_pos[:7], grip[:1] if grip.size else np.zeros(1, dtype=np.float32)]).astype(np.float32)
        elif action_space == 'joint_velocity':
            raw = np.asarray(step.get('action_dict', {}).get('joint_velocity', np.zeros(7, dtype=np.float32)), dtype=np.float32).reshape(-1)
            if raw.size < 7:
                raw = np.pad(raw, (0, 7 - raw.size))
            act = np.concatenate([raw[:7], grip[:1] if grip.size else np.zeros(1, dtype=np.float32)]).astype(np.float32)
        else:
            raw = np.asarray(step.get('action', np.zeros(7, dtype=np.float32)), dtype=np.float32).reshape(-1)
            act = raw.astype(np.float32)
        actions.append(act)
        rewards.append(float(step.get('reward', 0.0)))
        dones.append(1 if bool(step.get('is_last', False)) else 0)
        uncertainty.append(np.array([float(step.get('policy_uncertainty', 0.0))], dtype=np.float32))
        if image is None:
            image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        images.append(image)

    if not images:
        raise DroidSourceError(f'Episode {episode_key} did not contain any usable steps after step_stride={step_stride}.')

    return {
        'episode_key': episode_key,
        'source_episode_index': raw_episode.get('episode_index'),
        'task': 'droid_manipulation',
        'instruction': instruction,
        'outcome': outcome,
        'failure_onset': failure_onset,
        'images': np.asarray(images, dtype=np.uint8),
        'eef': np.asarray(eef, dtype=np.float32),
        'gripper': np.asarray(gripper, dtype=np.float32),
        'joint': np.asarray(joint, dtype=np.float32),
        'actions': np.asarray(actions, dtype=np.float32),
        'rewards': np.asarray(rewards, dtype=np.float32),
        'dones': np.asarray(dones, dtype=np.int32),
        'uncertainty': np.asarray(uncertainty, dtype=np.float32),
    }


class _RobomimicShardSink:
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
        self._current_final_path: Path | None = None
        self._current_write_path: Path | None = None
        self._max_source_episode_index: int | None = None
        if resume_existing:
            self._resume_existing()

    def _final_path(self, shard_index: int) -> Path:
        return self.split_dir / f'{self.prefix}_{shard_index:04d}.hdf5'

    def _inprogress_path(self, shard_index: int) -> Path:
        return self.split_dir / f'{self.prefix}_{shard_index:04d}.inprogress.hdf5'

    def _corrupt_path(self, path: Path) -> Path:
        stamp = int(time.time())
        return path.with_name(f'{path.name}.corrupt.{stamp}')

    def _scan_file_resume_metadata(self, path: Path) -> tuple[int, int | None]:
        with h5py.File(path, 'r') as f:
            if 'data' not in f:
                return 0, None
            data = f['data']
            keys = sorted(data.keys())
            if not keys:
                return 0, None
            last = data[keys[-1]]
            src_idx = None
            if 'source_episode_index' in last.attrs:
                try:
                    src_idx = int(last.attrs['source_episode_index'])
                except Exception:
                    src_idx = None
            return len(keys), src_idx

    def _count_episodes_in_file(self, path: Path) -> int:
        count, _ = self._scan_file_resume_metadata(path)
        return count

    def _max_source_episode_index_in_file(self, path: Path) -> int | None:
        _, src_idx = self._scan_file_resume_metadata(path)
        return src_idx

    def _quarantine_corrupt_file(self, path: Path) -> None:
        corrupt_path = self._corrupt_path(path)
        path.replace(corrupt_path)
        print(f'[convert-droid] quarantined corrupt trailing shard: {path} -> {corrupt_path}', file=sys.stderr, flush=True)

    def _try_open_resumable_last(self, path: Path, *, final_path: Path, count: int) -> bool:
        try:
            current_file = h5py.File(path, 'a')
            current_group = current_file.require_group('data')
        except Exception:
            return False
        self._current_final_path = final_path
        self._current_write_path = path
        self._current_file = current_file
        self._current_data_group = current_group
        self._current_count = int(count)
        return True

    def _resume_existing(self) -> None:
        while True:
            shard_files = sorted(self.split_dir.glob(f'{self.prefix}_*.hdf5'))
            if not shard_files:
                return

            valid_shards: list[Path] = []
            counts: list[int] = []
            source_indices: list[int | None] = []
            trailing_quarantined = False

            for idx, shard_path in enumerate(shard_files):
                try:
                    count = self._count_episodes_in_file(shard_path)
                    max_source_idx = self._max_source_episode_index_in_file(shard_path)
                except Exception:
                    is_trailing = idx == len(shard_files) - 1
                    if is_trailing:
                        self._quarantine_corrupt_file(shard_path)
                        trailing_quarantined = True
                        break
                    raise DroidSourceError(
                        f'Existing shard {shard_path} is corrupt and is not the trailing shard. '
                        'Please inspect the dataset directory before resuming.'
                    )
                valid_shards.append(shard_path)
                counts.append(count)
                source_indices.append(max_source_idx)

            if trailing_quarantined:
                # Re-scan after removing the corrupt suffix; there may be more than one.
                continue

            if not valid_shards:
                return

            finalized = [p for p in valid_shards if not p.name.endswith('.inprogress.hdf5')]
            self.shard_paths = [str(p) for p in finalized]
            self._episode_index = int(sum(counts))
            retained_sources = [idx for idx in source_indices if idx is not None]
            self._max_source_episode_index = max(retained_sources) if retained_sources else None

            last = valid_shards[-1]
            last_count = counts[-1]
            stem_name = last.name.replace('.inprogress.hdf5', '').replace('.hdf5', '')
            try:
                last_idx = int(stem_name.rsplit('_', 1)[1])
            except Exception:
                last_idx = len(valid_shards) - 1
            self._shard_index = last_idx + 1

            if last.name.endswith('.inprogress.hdf5'):
                ok = self._try_open_resumable_last(last, final_path=self._final_path(last_idx), count=last_count)
                if ok:
                    return
                self._quarantine_corrupt_file(last)
                continue

            if last_count < self.episodes_per_shard:
                ok = self._try_open_resumable_last(last, final_path=last, count=last_count)
                if ok:
                    return
                self._quarantine_corrupt_file(last)
                continue

            self._current_count = 0
            return

    @property
    def total_episodes(self) -> int:
        return int(self._episode_index)

    @property
    def max_source_episode_index(self) -> int | None:
        return self._max_source_episode_index

    def _ensure_open(self) -> None:
        if self._current_file is not None and self._current_count < self.episodes_per_shard:
            return
        self.close()
        final_path = self._final_path(self._shard_index)
        write_path = self._inprogress_path(self._shard_index)
        self._current_file = h5py.File(write_path, 'w')
        self._current_data_group = self._current_file.create_group('data')
        self._current_count = 0
        self._current_final_path = final_path
        self._current_write_path = write_path
        self._shard_index += 1

    def _finalize_current_shard(self) -> None:
        if self._current_file is None or self._current_write_path is None or self._current_final_path is None:
            return
        self._current_file.flush()
        self._current_file.close()
        if self._current_write_path != self._current_final_path:
            self._current_write_path.replace(self._current_final_path)
        if str(self._current_final_path) not in self.shard_paths:
            self.shard_paths.append(str(self._current_final_path))
        self._current_file = None
        self._current_data_group = None
        self._current_count = 0
        self._current_write_path = None
        self._current_final_path = None

    def _flush_current_shard(self) -> None:
        if self._current_file is not None:
            self._current_file.flush()

    def _write_precomputed(self, obs: h5py.Group, next_obs: h5py.Group, episode: dict, compression: str | None) -> None:
        if 'precomputed_vector' not in episode:
            return
        obs.create_dataset('precomputed_vector', data=episode['precomputed_vector'].astype(np.float16), compression=compression)
        obs.create_dataset('precomputed_visual_embedding', data=episode['precomputed_visual_embedding'].astype(np.float16), compression=compression)
        obs.create_dataset('precomputed_action_one_hot', data=episode['precomputed_action_one_hot'].astype(np.float16))
        obs.create_dataset('precomputed_object_gripper_dist', data=episode['precomputed_object_gripper_dist'].astype(np.float32))
        obs.create_dataset('precomputed_object_target_dist', data=episode['precomputed_object_target_dist'].astype(np.float32))
        obs.create_dataset('precomputed_object_height', data=episode['precomputed_object_height'].astype(np.float32))
        obs.create_dataset('precomputed_visibility', data=episode['precomputed_visibility'].astype(np.float32))

        next_obs.create_dataset('precomputed_vector', data=np.concatenate([episode['precomputed_vector'][1:], episode['precomputed_vector'][-1:]], axis=0).astype(np.float16), compression=compression)
        next_obs.create_dataset('precomputed_visual_embedding', data=np.concatenate([episode['precomputed_visual_embedding'][1:], episode['precomputed_visual_embedding'][-1:]], axis=0).astype(np.float16), compression=compression)
        next_obs.create_dataset('precomputed_action_one_hot', data=np.concatenate([episode['precomputed_action_one_hot'][1:], episode['precomputed_action_one_hot'][-1:]], axis=0).astype(np.float16))
        next_obs.create_dataset('precomputed_object_gripper_dist', data=np.concatenate([episode['precomputed_object_gripper_dist'][1:], episode['precomputed_object_gripper_dist'][-1:]], axis=0).astype(np.float32))
        next_obs.create_dataset('precomputed_object_target_dist', data=np.concatenate([episode['precomputed_object_target_dist'][1:], episode['precomputed_object_target_dist'][-1:]], axis=0).astype(np.float32))
        next_obs.create_dataset('precomputed_object_height', data=np.concatenate([episode['precomputed_object_height'][1:], episode['precomputed_object_height'][-1:]], axis=0).astype(np.float32))
        next_obs.create_dataset('precomputed_visibility', data=np.concatenate([episode['precomputed_visibility'][1:], episode['precomputed_visibility'][-1:]], axis=0).astype(np.float32))

    def add_episode(self, episode: dict) -> None:
        self._ensure_open()
        assert self._current_data_group is not None
        demo = self._current_data_group.create_group(f'demo_{self._episode_index:06d}')
        self._episode_index += 1
        self._current_count += 1

        compression = self.compression
        demo.attrs['task'] = episode['task']
        demo.attrs['instruction'] = episode['instruction']
        demo.attrs['outcome'] = episode['outcome']
        demo.attrs['source_format'] = 'droid_rlds'
        demo.attrs['episode_key'] = episode['episode_key']
        if episode.get('source_episode_index') is not None:
            demo.attrs['source_episode_index'] = int(episode['source_episode_index'])
            self._max_source_episode_index = int(episode['source_episode_index'])
        if episode['failure_onset'] is not None:
            demo.attrs['failure_onset'] = int(episode['failure_onset'])

        demo.create_dataset('actions', data=episode['actions'].astype(np.float32))
        demo.create_dataset('rewards', data=episode['rewards'].astype(np.float32))
        demo.create_dataset('dones', data=episode['dones'].astype(np.int32))
        obs = demo.create_group('obs')
        next_obs = demo.create_group('next_obs')
        obs.create_dataset('agentview_image', data=episode['images'].astype(np.uint8), compression=compression)
        obs.create_dataset('robot0_eef_pos', data=episode['eef'].astype(np.float32))
        obs.create_dataset('robot0_gripper_qpos', data=episode['gripper'].astype(np.float32))
        obs.create_dataset('robot0_joint_pos', data=episode['joint'].astype(np.float32))
        obs.create_dataset('object_pos', data=np.zeros_like(episode['eef'], dtype=np.float32))
        obs.create_dataset('goal_pos', data=np.zeros_like(episode['eef'], dtype=np.float32))
        obs.create_dataset('policy_uncertainty', data=episode['uncertainty'].astype(np.float32))

        next_images = np.concatenate([episode['images'][1:], episode['images'][-1:]], axis=0)
        next_eef = np.concatenate([episode['eef'][1:], episode['eef'][-1:]], axis=0)
        next_gripper = np.concatenate([episode['gripper'][1:], episode['gripper'][-1:]], axis=0)
        next_joint = np.concatenate([episode['joint'][1:], episode['joint'][-1:]], axis=0)
        next_unc = np.concatenate([episode['uncertainty'][1:], episode['uncertainty'][-1:]], axis=0)
        next_obs.create_dataset('agentview_image', data=next_images.astype(np.uint8), compression=compression)
        next_obs.create_dataset('robot0_eef_pos', data=next_eef.astype(np.float32))
        next_obs.create_dataset('robot0_gripper_qpos', data=next_gripper.astype(np.float32))
        next_obs.create_dataset('robot0_joint_pos', data=next_joint.astype(np.float32))
        next_obs.create_dataset('object_pos', data=np.zeros_like(next_eef, dtype=np.float32))
        next_obs.create_dataset('goal_pos', data=np.zeros_like(next_eef, dtype=np.float32))
        next_obs.create_dataset('policy_uncertainty', data=next_unc.astype(np.float32))
        self._write_precomputed(obs, next_obs, episode, compression)
        self._flush_current_shard()
        if self._current_count >= self.episodes_per_shard:
            self._finalize_current_shard()

    def close(self) -> None:
        if self._current_file is not None:
            self._finalize_current_shard()


def _build_conversion_encoder(encoder_name: str | None, device: str | None):
    if encoder_name is None or str(encoder_name).lower() in {'', 'none', 'off'}:
        return None
    cfg = AppConfig()
    cfg.encoder.name = str(encoder_name)
    if device:
        cfg.encoder.device = str(device)
    return build_encoder(cfg.encoder)


def _attach_precomputed_features(episode: dict, encoder) -> dict:
    if encoder is None:
        return episode
    vectors = []
    visuals = []
    action_one_hot = []
    object_gripper_dist = []
    object_target_dist = []
    object_height = []
    visibility = []
    prev_gripper = None
    zeros = np.zeros(3, dtype=np.float32)
    observations: list[StepObservation] = []
    batch_size = max(1, int(getattr(getattr(encoder, 'config', None), 'convert_batch_size', 16) or 16))

    def _flush(batch: list[StepObservation]) -> None:
        if not batch:
            return
        if hasattr(encoder, 'extract_batch'):
            snaps = encoder.extract_batch(batch)
        else:
            snaps = [encoder.extract(obs) for obs in batch]
        for snap in snaps:
            vectors.append(np.asarray(snap.vector, dtype=np.float32))
            visuals.append(np.asarray(snap.visual_embedding, dtype=np.float32))
            action_one_hot.append(np.asarray(snap.action_one_hot, dtype=np.float32))
            object_gripper_dist.append(np.array([snap.object_gripper_dist], dtype=np.float32))
            object_target_dist.append(np.array([snap.object_target_dist], dtype=np.float32))
            object_height.append(np.array([snap.object_height], dtype=np.float32))
            visibility.append(np.array([snap.visibility], dtype=np.float32))

    for i in range(len(episode['images'])):
        grip = float(np.asarray(episode['gripper'][i]).reshape(-1)[0])
        action = np.asarray(episode['actions'][i], dtype=np.float32).reshape(-1)
        action_type = infer_action_type(action, grip, prev_gripper)
        prev_gripper = grip
        robot_state = np.concatenate([
            np.asarray(episode['eef'][i], dtype=np.float32).reshape(-1)[:3],
            np.array([grip], dtype=np.float32),
            zeros,
            zeros,
            np.asarray(episode['joint'][i], dtype=np.float32).reshape(-1),
        ], dtype=np.float32)
        policy_uncertainty = float(np.asarray(episode['uncertainty'][i]).reshape(-1)[0])
        policy_stats = np.array([
            float(min(1.0, np.linalg.norm(action))),
            float(abs(action[-1]) if len(action) else 0.0),
            1.0,
            policy_uncertainty,
        ], dtype=np.float32)
        observations.append(
            StepObservation(
                image=np.asarray(episode['images'][i], dtype=np.uint8),
                robot_state=robot_state,
                action=action,
                policy_stats=policy_stats,
                action_type=action_type,
                timestamp=i,
                instruction=episode['instruction'],
            )
        )
        if len(observations) >= batch_size:
            _flush(observations)
            observations = []
    _flush(observations)
    episode['precomputed_vector'] = np.stack(vectors, axis=0)
    episode['precomputed_visual_embedding'] = np.stack(visuals, axis=0)
    episode['precomputed_action_one_hot'] = np.stack(action_one_hot, axis=0)
    episode['precomputed_object_gripper_dist'] = np.stack(object_gripper_dist, axis=0)
    episode['precomputed_object_target_dist'] = np.stack(object_target_dist, axis=0)
    episode['precomputed_object_height'] = np.stack(object_height, axis=0)
    episode['precomputed_visibility'] = np.stack(visibility, axis=0)
    return episode


def _load_conversion_state(output_dir: Path, *, source_label: str, source_mode: str, episodes_per_shard: int, image_size: int, step_stride: int, action_space: str, outcome_filter: str = 'all') -> dict:
    state_path = output_dir / '.convert_state.json'
    if not state_path.exists():
        return {
            'processed_total': 0,
            'counts': {'train': 0, 'calib': 0, 'eval': 0},
            'completed': False,
            'source': source_label,
            'source_mode': source_mode,
            'episodes_per_shard': episodes_per_shard,
            'image_size': image_size,
            'step_stride': step_stride,
            'action_space': action_space,
            'outcome_filter': outcome_filter,
        }
    state = load_json(state_path)
    expected = {
        'source': source_label,
        'source_mode': source_mode,
        'episodes_per_shard': episodes_per_shard,
        'image_size': image_size,
        'step_stride': step_stride,
        'action_space': action_space,
        'outcome_filter': outcome_filter,
    }
    for k, v in expected.items():
        if state.get(k) != v:
            raise DroidSourceError(f'Existing conversion checkpoint at {state_path} does not match current settings: {k}={state.get(k)!r} != {v!r}')
    return state


def _write_conversion_state(output_dir: Path, payload: dict) -> None:
    atomic_write_json(output_dir / '.convert_state.json', payload)


def convert_droid_source_to_shards(
    source: DroidEpisodeSource,
    output_dir: str | Path,
    *,
    source_label: str,
    source_mode: str,
    episodes_per_shard: int = 64,
    image_size: int = 96,
    step_stride: int = 1,
    action_space: str = 'raw_action',
    camera_preference: tuple[str, ...] = DEFAULT_CAMERA_PREFERENCE,
    compression: str | None = 'gzip',
    show_progress: bool = True,
    resume: bool = True,
    checkpoint_every: int = 32,
    precompute_encoder: str | None = None,
    precompute_device: str | None = None,
    outcome_filter: str = 'all',
) -> DroidShardManifest:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outcome_filter = str(outcome_filter or 'all').lower()
    if outcome_filter not in {'all', 'success', 'failure'}:
        raise ValueError(f'Unsupported outcome_filter={outcome_filter!r}; expected all, success, or failure.')

    manifest_path = output_dir / 'manifest.json'

    total = None
    if hasattr(source, 'estimate_num_episodes'):
        try:
            total = source.estimate_num_episodes()
        except Exception:
            total = None

    if resume and manifest_path.exists() and not (output_dir / '.convert_state.json').exists() and outcome_filter == 'all':
        return DroidShardManifest(**load_json(manifest_path))

    state = _load_conversion_state(
        output_dir,
        source_label=source_label,
        source_mode=source_mode,
        episodes_per_shard=episodes_per_shard,
        image_size=image_size,
        step_stride=step_stride,
        action_space=action_space,
        outcome_filter=outcome_filter,
    ) if resume else {
        'processed_total': 0,
        'counts': {'train': 0, 'calib': 0, 'eval': 0},
        'completed': False,
        'outcome_filter': outcome_filter,
    }

    sinks = {
        'train': _RobomimicShardSink(output_dir / 'train', 'droid_train', episodes_per_shard, compression=compression, resume_existing=resume),
        'calib': _RobomimicShardSink(output_dir / 'calib', 'droid_calib', episodes_per_shard, compression=compression, resume_existing=resume),
        'eval': _RobomimicShardSink(output_dir / 'eval', 'droid_eval', episodes_per_shard, compression=compression, resume_existing=resume),
    }
    counts = {split: sinks[split].total_episodes for split in ('train', 'calib', 'eval')}
    retained_source_indices = [
        sinks[split].max_source_episode_index for split in ('train', 'calib', 'eval')
        if sinks[split].max_source_episode_index is not None
    ]
    retained_resume_from = (max(retained_source_indices) + 1) if retained_source_indices else None
    state_processed = int(state.get('processed_total', int(sum(counts.values()))))
    if retained_resume_from is not None:
        processed = min(state_processed, retained_resume_from)
    elif outcome_filter == 'all':
        processed = int(sum(counts.values()))
    else:
        processed = state_processed

    if state.get('completed') and manifest_path.exists():
        state_processed = int(state.get('processed_total', 0))
        if total is not None and state_processed >= total:
            return DroidShardManifest(**load_json(manifest_path))

    progress = ETATracker(
        label='convert-droid',
        total=total,
        unit='episodes',
        print_every=max(1, episodes_per_shard // 2),
        initial_current=processed,
    ) if show_progress else None
    if progress is not None and processed > 0:
        progress.update(
            processed,
            extra=(f"resume_from={processed} train={counts['train']} calib={counts['calib']} eval={counts['eval']}"),
            force=True,
        )

    encoder = _build_conversion_encoder(precompute_encoder, precompute_device)

    for raw_episode in source.iter_episodes(start_episode=processed):
        episode = _standardize_droid_episode(
            raw_episode,
            image_size=image_size,
            step_stride=step_stride,
            action_space=action_space,
            camera_preference=camera_preference,
        )
        processed += 1
        if outcome_filter != 'all' and str(episode['outcome']).lower() != outcome_filter:
            if progress is not None:
                progress.update(
                    processed,
                    extra=(
                        f"kept_train={counts['train']} kept_calib={counts['calib']} kept_eval={counts['eval']} "
                        f"skipped_outcome={episode['outcome']} last_episode={episode['episode_key']}"
                    ),
                )
            if checkpoint_every > 0 and processed % checkpoint_every == 0:
                _write_conversion_state(output_dir, {
                    'processed_total': processed,
                    'counts': counts,
                    'completed': False,
                    'source': source_label,
                    'source_mode': source_mode,
                    'episodes_per_shard': episodes_per_shard,
                    'image_size': image_size,
                    'step_stride': step_stride,
                    'action_space': action_space,
                    'outcome_filter': outcome_filter,
                    'train_shards': sinks['train'].shard_paths,
                    'calib_shards': sinks['calib'].shard_paths,
                    'eval_shards': sinks['eval'].shard_paths,
                    'last_episode_key': episode['episode_key'],
                    'precompute_encoder': precompute_encoder,
                    'precompute_device': precompute_device,
                })
            continue
        episode = _attach_precomputed_features(episode, encoder)
        split = _hash_to_split(episode['episode_key'])
        sinks[split].add_episode(episode)
        counts[split] += 1
        if progress is not None:
            progress.update(
                processed,
                extra=(
                    f"train={counts['train']} calib={counts['calib']} eval={counts['eval']} "
                    f"last_split={split} last_episode={episode['episode_key']}"
                ),
            )
        if checkpoint_every > 0 and processed % checkpoint_every == 0:
            _write_conversion_state(output_dir, {
                'processed_total': processed,
                'counts': counts,
                'completed': False,
                'source': source_label,
                'source_mode': source_mode,
                'episodes_per_shard': episodes_per_shard,
                'image_size': image_size,
                'step_stride': step_stride,
                'action_space': action_space,
                'outcome_filter': outcome_filter,
                'train_shards': sinks['train'].shard_paths,
                'calib_shards': sinks['calib'].shard_paths,
                'eval_shards': sinks['eval'].shard_paths,
                'last_episode_key': episode['episode_key'],
                'precompute_encoder': precompute_encoder,
                'precompute_device': precompute_device,
            })

    for sink in sinks.values():
        sink.close()
    if progress is not None:
        progress.done(
            current=processed,
            extra=(
                f"finished | train={counts['train']} calib={counts['calib']} eval={counts['eval']}"
            ),
        )

    manifest = DroidShardManifest(
        source=source_label,
        source_mode=source_mode,
        train_shards=sinks['train'].shard_paths,
        calib_shards=sinks['calib'].shard_paths,
        eval_shards=sinks['eval'].shard_paths,
        train_episodes=counts['train'],
        calib_episodes=counts['calib'],
        eval_episodes=counts['eval'],
        image_size=image_size,
        step_stride=step_stride,
        episodes_per_shard=episodes_per_shard,
        action_space=action_space,
    )
    atomic_write_json(manifest_path, manifest.as_dict())
    _write_conversion_state(output_dir, {
        'processed_total': processed,
        'counts': counts,
        'completed': True,
        'source': source_label,
        'source_mode': source_mode,
        'episodes_per_shard': episodes_per_shard,
        'image_size': image_size,
        'step_stride': step_stride,
        'action_space': action_space,
        'outcome_filter': outcome_filter,
        'train_shards': sinks['train'].shard_paths,
        'calib_shards': sinks['calib'].shard_paths,
        'eval_shards': sinks['eval'].shard_paths,
        'precompute_encoder': precompute_encoder,
        'precompute_device': precompute_device,
    })
    return manifest


def create_mock_droid_shards(
    output_dir: str | Path,
    *,
    num_episodes: int = 18,
    steps_per_episode: int = 20,
    episodes_per_shard: int = 4,
    image_size: int = 96,
    include_failures: bool = True,
    seed: int = 5,
) -> DroidShardManifest:
    source = MockDroidEpisodeSource(
        num_episodes=num_episodes,
        steps_per_episode=steps_per_episode,
        image_size=image_size,
        include_failures=include_failures,
        seed=seed,
    )
    return convert_droid_source_to_shards(
        source,
        output_dir,
        source_label='mock_droid',
        source_mode='mock',
        episodes_per_shard=episodes_per_shard,
        image_size=image_size,
        step_stride=1,
        action_space='raw_action',
    )
