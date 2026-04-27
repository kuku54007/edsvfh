from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import AppConfig, EncoderConfig
from .encoders import build_encoder
from .fino_convert import _load_manifest, _standardize_manifest_episode, convert_failure_manifest_to_shards
from .progress import ETATracker
from .public_data import infer_action_type, list_hdf5_shards, load_robomimic_hdf5
from .types import Episode, EpisodeStep, StepObservation
from .fino_finetune import fine_tune_bundle_on_failure_shards

_EPS = 1e-6


class FIPERPseudoOnsetError(RuntimeError):
    pass


@dataclass(frozen=True)
class FIPERStyleNormalBaseline:
    encoder: str
    feature_source: str
    phase_feature_mean: np.ndarray
    phase_feature_std: np.ndarray
    phase_feature_count: np.ndarray
    global_feature_mean: np.ndarray
    global_feature_std: np.ndarray
    obs_thresholds: np.ndarray
    uncertainty_thresholds: np.ndarray
    global_obs_threshold: float
    global_uncertainty_threshold: float
    window: int
    phase_bins: int
    quantile: float
    min_phase_count: int
    num_success_episodes: int
    num_success_steps: int
    metadata: dict[str, Any]

    def describe(self) -> dict[str, Any]:
        return {
            'encoder': self.encoder,
            'feature_source': self.feature_source,
            'feature_dim': int(self.global_feature_mean.shape[0]),
            'window': int(self.window),
            'phase_bins': int(self.phase_bins),
            'quantile': float(self.quantile),
            'min_phase_count': int(self.min_phase_count),
            'num_success_episodes': int(self.num_success_episodes),
            'num_success_steps': int(self.num_success_steps),
            'global_obs_threshold': float(self.global_obs_threshold),
            'global_uncertainty_threshold': float(self.global_uncertainty_threshold),
            **self.metadata,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> 'FIPERStyleNormalBaseline':
        with Path(path).open('rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f'Unexpected baseline type: {type(obj)!r}')
        return obj


@dataclass(frozen=True)
class FIPERPseudoOnsetEpisodeResult:
    pseudo_failure_onset: int | None
    reason: str
    confidence: float
    trigger_index: int | None
    obs_scores: np.ndarray
    uncertainty_scores: np.ndarray
    obs_thresholds: np.ndarray
    uncertainty_thresholds: np.ndarray
    trigger_mask: np.ndarray


@dataclass(frozen=True)
class PseudoOnsetManifestResult:
    baseline_path: str
    output_path: str
    total_episodes: int
    failure_episodes: int
    pseudo_labeled_failures: int
    success_episodes: int
    replaced_failure_onsets: int
    preserved_original_failure_onsets: int
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class PseudoOnsetRebuildResult:
    baseline_path: str
    pseudo_manifest_path: str
    converted_root: str
    output_bundle: str
    metrics: dict[str, float]
    strategies: list[str]


class _RunningArrayStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum: np.ndarray | None = None
        self.sumsq: np.ndarray | None = None

    def update(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if self.sum is None:
            self.sum = np.zeros_like(arr, dtype=np.float64)
            self.sumsq = np.zeros_like(arr, dtype=np.float64)
        if arr.shape != self.sum.shape:
            raise FIPERPseudoOnsetError(f'Feature dimension mismatch: {arr.shape} != {self.sum.shape}')
        self.sum += arr
        self.sumsq += arr * arr
        self.count += 1

    def mean_std(self, *, default_dim: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if self.count <= 0:
            if default_dim is None:
                raise FIPERPseudoOnsetError('Cannot finalize empty running statistics without default_dim.')
            return np.zeros((default_dim,), dtype=np.float32), np.ones((default_dim,), dtype=np.float32)
        assert self.sum is not None and self.sumsq is not None
        mean = self.sum / float(self.count)
        var = np.maximum(self.sumsq / float(self.count) - mean * mean, _EPS)
        std = np.sqrt(var)
        return mean.astype(np.float32), std.astype(np.float32)


def _phase_bin(step_idx: int, length: int, phase_bins: int) -> int:
    if phase_bins <= 1 or length <= 1:
        return 0
    ratio = float(step_idx) / float(max(1, length - 1))
    return min(phase_bins - 1, max(0, int(ratio * phase_bins)))


def _is_success_episode(episode: Episode) -> bool:
    outcome = str(episode.outcome or '').lower()
    return outcome not in {'failure', 'fail'} and episode.failure_onset is None


def _deduplicate_shards(shard_dir: str | Path) -> list[Path]:
    root = Path(shard_dir)
    paths: list[Path] = []
    if root.is_file():
        return [root]
    for split in ('train', 'calib', 'eval'):
        paths.extend(list_hdf5_shards(root, split=split))
    if not paths:
        paths.extend(list_hdf5_shards(root))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _feature_from_snapshot(snapshot, *, feature_source: str) -> np.ndarray:
    if feature_source == 'visual':
        feat = np.asarray(snapshot.visual_embedding, dtype=np.float32).reshape(-1)
    elif feature_source == 'vector':
        feat = np.asarray(snapshot.vector, dtype=np.float32).reshape(-1)
    else:
        raise ValueError(f'Unsupported feature_source={feature_source!r}; expected visual or vector.')
    if feat.size == 0:
        raise FIPERPseudoOnsetError('Selected feature_source produced an empty feature vector.')
    return feat


def _action_uncertainty_proxy(obs: StepObservation) -> float:
    explicit = 0.0
    if len(obs.policy_stats) >= 4:
        explicit = float(max(0.0, obs.policy_stats[3]))
    action = np.asarray(obs.action, dtype=np.float32).reshape(-1)
    if action.size == 0:
        motion_proxy = 0.0
    else:
        move = float(np.linalg.norm(action[:3])) if action.size >= 3 else float(np.linalg.norm(action))
        grip = float(abs(action[-1]))
        motion_proxy = min(1.0, 0.25 * move + 0.50 * grip)
    return float(explicit + motion_proxy)


def _visual_change_uncertainty_proxy(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    proxy = np.zeros((features.shape[0],), dtype=np.float32)
    if features.shape[0] <= 1:
        return proxy
    deltas = np.linalg.norm(np.diff(features, axis=0), axis=1).astype(np.float32)
    scale = float(np.quantile(deltas, 0.90)) if deltas.size else 0.0
    if scale <= _EPS:
        scale = float(np.max(deltas)) if deltas.size else 0.0
    if scale > _EPS:
        deltas = np.clip(deltas / scale, 0.0, 1.0)
    else:
        deltas = np.zeros_like(deltas, dtype=np.float32)
    proxy[1:] = deltas
    return proxy


def _episode_feature_matrix(episode: Episode, encoder, *, feature_source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations = [step.observation for step in episode.steps]
    if not observations:
        raise FIPERPseudoOnsetError('Episode does not contain any steps.')
    if hasattr(encoder, 'extract_batch'):
        snapshots = encoder.extract_batch(observations)
    else:
        snapshots = [encoder.extract(obs) for obs in observations]
    features = np.stack([_feature_from_snapshot(snapshot, feature_source=feature_source) for snapshot in snapshots], axis=0).astype(np.float32)
    uncertainty = np.asarray([_action_uncertainty_proxy(obs) for obs in observations], dtype=np.float32)
    if uncertainty.size > 0 and float(np.max(uncertainty)) <= _EPS:
        uncertainty = np.maximum(uncertainty, _visual_change_uncertainty_proxy(features))
    phase_bins = np.asarray([_phase_bin(idx, len(observations), 1) for idx in range(len(observations))], dtype=np.int64)
    return features, uncertainty, phase_bins


def _phase_indices(length: int, phase_bins: int) -> np.ndarray:
    return np.asarray([_phase_bin(idx, length, phase_bins) for idx in range(length)], dtype=np.int64)


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr.astype(np.float32)
    if window <= 1:
        return arr.astype(np.float32)
    out = np.zeros_like(arr, dtype=np.float32)
    cumsum = np.cumsum(arr, dtype=np.float64)
    for idx in range(arr.size):
        start = max(0, idx - window + 1)
        total = cumsum[idx] - (cumsum[start - 1] if start > 0 else 0.0)
        out[idx] = float(total)
    return out


def _episode_obs_scores(features: np.ndarray, phase_idx: np.ndarray, baseline: FIPERStyleNormalBaseline) -> np.ndarray:
    scores = np.zeros((features.shape[0],), dtype=np.float32)
    for idx, feat in enumerate(features):
        phase = int(phase_idx[idx])
        if baseline.phase_feature_count[phase] >= baseline.min_phase_count:
            mean = baseline.phase_feature_mean[phase]
            std = baseline.phase_feature_std[phase]
        else:
            mean = baseline.global_feature_mean
            std = baseline.global_feature_std
        z = np.abs((feat - mean) / np.maximum(std, _EPS))
        z = np.clip(z, 0.0, 8.0)
        scores[idx] = float(np.sqrt(np.mean(z * z)))
    return scores


def _quantile_or_default(values: list[float], quantile: float, default: float) -> float:
    if not values:
        return float(default)
    return float(np.quantile(np.asarray(values, dtype=np.float32), quantile))


def _write_manifest_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == '.jsonl':
        with path.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        return
    with path.open('w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def _runtime_episode_from_manifest_entry(entry: dict[str, Any], *, image_size: int) -> Episode:
    standardized = _standardize_manifest_episode(entry, image_size=image_size)
    images = np.asarray(standardized['images'], dtype=np.uint8)
    eef = np.asarray(standardized['eef'], dtype=np.float32)
    gripper = np.asarray(standardized['gripper'], dtype=np.float32)
    object_pos = np.asarray(standardized['object_pos'], dtype=np.float32)
    goal_pos = np.asarray(standardized['goal_pos'], dtype=np.float32)
    actions = np.asarray(standardized['actions'], dtype=np.float32)
    uncertainty = np.asarray(standardized['uncertainty'], dtype=np.float32)
    steps: list[EpisodeStep] = []
    prev_gripper: float | None = None
    for idx in range(images.shape[0]):
        grip_arr = np.asarray(gripper[idx], dtype=np.float32).reshape(-1)
        grip = float(grip_arr[0]) if grip_arr.size else 0.0
        action = np.asarray(actions[idx], dtype=np.float32).reshape(-1)
        action_type = infer_action_type(action, grip, prev_gripper)
        prev_gripper = grip
        policy_uncertainty = float(np.asarray(uncertainty[idx], dtype=np.float32).reshape(-1)[0]) if uncertainty.size else 0.0
        policy_stats = np.array([
            float(min(1.0, np.linalg.norm(action))),
            float(abs(action[-1]) if action.size else 0.0),
            1.0,
            policy_uncertainty,
        ], dtype=np.float32)
        robot_state = np.concatenate([
            np.asarray(eef[idx], dtype=np.float32).reshape(-1)[:3],
            np.array([grip], dtype=np.float32),
            np.asarray(object_pos[idx], dtype=np.float32).reshape(-1)[:3],
            np.asarray(goal_pos[idx], dtype=np.float32).reshape(-1)[:3],
        ], dtype=np.float32)
        obs = StepObservation(
            image=np.asarray(images[idx], dtype=np.uint8),
            robot_state=robot_state,
            action=action,
            policy_stats=policy_stats,
            action_type=action_type,
            timestamp=idx,
            instruction=standardized['instruction'],
        )
        steps.append(EpisodeStep(observation=obs))
    return Episode(
        task=str(standardized['task']),
        instruction=str(standardized['instruction']),
        steps=steps,
        outcome=str(standardized['outcome']),
        failure_onset=standardized['failure_onset'],
        source=str(entry.get('frames_dir', entry.get('episode_id', 'fino_manifest'))),
    )


def fit_droid_success_baseline(
    shard_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    config: AppConfig | None = None,
    feature_source: str = 'visual',
    window: int = 3,
    phase_bins: int = 10,
    quantile: float = 0.97,
    min_phase_count: int = 8,
    max_episodes: int | None = None,
    show_progress: bool = True,
) -> FIPERStyleNormalBaseline:
    if not (0.5 < quantile < 1.0):
        raise ValueError('quantile must satisfy 0.5 < quantile < 1.0')
    if window < 1:
        raise ValueError('window must be >= 1')
    if phase_bins < 1:
        raise ValueError('phase_bins must be >= 1')
    config = config or AppConfig()
    shards = _deduplicate_shards(shard_dir)
    if not shards:
        raise FileNotFoundError(f'No robomimic shards found under {Path(shard_dir)}')
    encoder = build_encoder(config.encoder)

    phase_stats = [_RunningArrayStats() for _ in range(phase_bins)]
    global_stats = _RunningArrayStats()
    success_episodes = 0
    success_steps = 0
    processed_success_shards: list[str] = []
    progress = ETATracker(label='fit-success-baseline:stats', total=max_episodes, unit='episodes', print_every=1) if show_progress else None

    for shard_path in shards:
        episodes = load_robomimic_hdf5(shard_path, config=config.dataset)
        shard_used = False
        for episode in episodes:
            if max_episodes is not None and success_episodes >= max_episodes:
                break
            if not _is_success_episode(episode):
                continue
            features, _uncertainty, _ = _episode_feature_matrix(episode, encoder, feature_source=feature_source)
            phase_idx = _phase_indices(features.shape[0], phase_bins)
            for feat, phase in zip(features, phase_idx):
                phase_stats[int(phase)].update(feat)
                global_stats.update(feat)
                success_steps += 1
            success_episodes += 1
            shard_used = True
            if progress is not None:
                progress.update(success_episodes, extra=f'shard={Path(shard_path).name} success_steps={success_steps}')
        if shard_used:
            processed_success_shards.append(str(shard_path))
        if max_episodes is not None and success_episodes >= max_episodes:
            break

    if success_episodes <= 0:
        raise FIPERPseudoOnsetError('Could not find any success episodes to build the normal baseline.')

    global_mean, global_std = global_stats.mean_std()
    phase_feature_mean = np.zeros((phase_bins, global_mean.shape[0]), dtype=np.float32)
    phase_feature_std = np.zeros((phase_bins, global_mean.shape[0]), dtype=np.float32)
    phase_feature_count = np.zeros((phase_bins,), dtype=np.int64)
    for phase, stats in enumerate(phase_stats):
        phase_feature_count[phase] = int(stats.count)
        if stats.count >= min_phase_count:
            mean, std = stats.mean_std(default_dim=global_mean.shape[0])
        else:
            mean, std = global_mean.copy(), global_std.copy()
        phase_feature_mean[phase] = mean
        phase_feature_std[phase] = np.maximum(std, _EPS)

    obs_values_by_phase: list[list[float]] = [[] for _ in range(phase_bins)]
    uncertainty_values_by_phase: list[list[float]] = [[] for _ in range(phase_bins)]
    global_obs_values: list[float] = []
    global_uncertainty_values: list[float] = []

    baseline_stub = FIPERStyleNormalBaseline(
        encoder=config.encoder.name,
        feature_source=feature_source,
        phase_feature_mean=phase_feature_mean,
        phase_feature_std=phase_feature_std,
        phase_feature_count=phase_feature_count,
        global_feature_mean=global_mean,
        global_feature_std=np.maximum(global_std, _EPS),
        obs_thresholds=np.zeros((phase_bins,), dtype=np.float32),
        uncertainty_thresholds=np.zeros((phase_bins,), dtype=np.float32),
        global_obs_threshold=0.0,
        global_uncertainty_threshold=0.0,
        window=window,
        phase_bins=phase_bins,
        quantile=quantile,
        min_phase_count=min_phase_count,
        num_success_episodes=success_episodes,
        num_success_steps=success_steps,
        metadata={'source': str(shard_dir), 'processed_success_shards': processed_success_shards},
    )

    progress = ETATracker(label='fit-success-baseline:thresholds', total=success_episodes, unit='episodes', print_every=1) if show_progress else None
    rescored_episodes = 0
    for shard_path in processed_success_shards:
        episodes = load_robomimic_hdf5(shard_path, config=config.dataset)
        for episode in episodes:
            if not _is_success_episode(episode):
                continue
            features, uncertainty, _ = _episode_feature_matrix(episode, encoder, feature_source=feature_source)
            phase_idx = _phase_indices(features.shape[0], phase_bins)
            raw_obs = _episode_obs_scores(features, phase_idx, baseline_stub)
            agg_obs = _rolling_sum(raw_obs, window)
            agg_uncertainty = _rolling_sum(uncertainty, window)
            for idx, phase in enumerate(phase_idx):
                obs_val = float(agg_obs[idx])
                unc_val = float(agg_uncertainty[idx])
                obs_values_by_phase[int(phase)].append(obs_val)
                uncertainty_values_by_phase[int(phase)].append(unc_val)
                global_obs_values.append(obs_val)
                global_uncertainty_values.append(unc_val)
            rescored_episodes += 1
            if progress is not None:
                progress.update(rescored_episodes, extra=f'shard={Path(shard_path).name}')
            if max_episodes is not None and rescored_episodes >= success_episodes:
                break
        if max_episodes is not None and rescored_episodes >= success_episodes:
            break

    global_obs_threshold = _quantile_or_default(global_obs_values, quantile, 1.0)
    global_uncertainty_threshold = _quantile_or_default(global_uncertainty_values, quantile, 0.1)
    obs_thresholds = np.asarray([
        _quantile_or_default(values, quantile, global_obs_threshold)
        for values in obs_values_by_phase
    ], dtype=np.float32)
    uncertainty_thresholds = np.asarray([
        _quantile_or_default(values, quantile, global_uncertainty_threshold)
        for values in uncertainty_values_by_phase
    ], dtype=np.float32)

    baseline = FIPERStyleNormalBaseline(
        encoder=config.encoder.name,
        feature_source=feature_source,
        phase_feature_mean=phase_feature_mean,
        phase_feature_std=np.maximum(phase_feature_std, _EPS),
        phase_feature_count=phase_feature_count,
        global_feature_mean=global_mean,
        global_feature_std=np.maximum(global_std, _EPS),
        obs_thresholds=np.maximum(obs_thresholds, _EPS),
        uncertainty_thresholds=np.maximum(uncertainty_thresholds, _EPS),
        global_obs_threshold=max(float(global_obs_threshold), _EPS),
        global_uncertainty_threshold=max(float(global_uncertainty_threshold), _EPS),
        window=window,
        phase_bins=phase_bins,
        quantile=quantile,
        min_phase_count=min_phase_count,
        num_success_episodes=success_episodes,
        num_success_steps=success_steps,
        metadata={'source': str(shard_dir), 'processed_success_shards': processed_success_shards},
    )
    if output_path is not None:
        baseline.save(output_path)
    return baseline


def _copy_encoder_config(config: AppConfig, *, name: str) -> EncoderConfig:
    encoder_cfg = config.encoder.model_copy(deep=True)
    encoder_cfg.name = str(name)
    return encoder_cfg


def _encoder_name_hints(expected_dim: int) -> list[str]:
    hints: list[str] = []
    if expected_dim == 51:
        hints.append('fallback')
    if expected_dim >= 1500:
        hints.append('siglip2_dinov2')
    if 760 <= expected_dim <= 900:
        hints.append('dinov2')
    return hints


def _encoder_candidate_names(config: AppConfig, baseline: FIPERStyleNormalBaseline) -> list[str]:
    expected_dim = int(baseline.global_feature_mean.shape[0])
    raw = [
        getattr(config.encoder, 'name', None),
        baseline.encoder,
        *(_encoder_name_hints(expected_dim)),
        'siglip2_dinov2',
        'dinov2',
        'fallback',
    ]
    names: list[str] = []
    seen: set[str] = set()
    for name in raw:
        if name is None:
            continue
        key = str(name).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        names.append(str(name))
    return names


def _encoder_debug_name(encoder: Any) -> str:
    cfg = getattr(encoder, 'config', None)
    name = getattr(cfg, 'name', None)
    if name is not None:
        return str(name)
    return encoder.__class__.__name__


def _resolve_episode_features_for_baseline(
    episode: Episode,
    baseline: FIPERStyleNormalBaseline,
    *,
    config: AppConfig,
    encoder=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    expected_dim = int(baseline.global_feature_mean.shape[0])
    tried_dims: list[str] = []
    build_errors: list[str] = []

    def _attempt_with_encoder(active_encoder: Any, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any] | None:
        try:
            features, uncertainty, _ = _episode_feature_matrix(episode, active_encoder, feature_source=baseline.feature_source)
        except Exception as exc:
            build_errors.append(f'{label}: extract failed ({exc})')
            return None
        actual_dim = int(features.shape[1])
        tried_dims.append(f'{label}={actual_dim}')
        if actual_dim != expected_dim:
            return None
        phase_idx = _phase_indices(features.shape[0], baseline.phase_bins)
        return features, uncertainty, phase_idx, active_encoder

    initial = encoder
    if initial is None:
        try:
            initial = build_encoder(config.encoder)
        except Exception as exc:
            build_errors.append(f'{getattr(config.encoder, "name", "<unset>")}: build failed ({exc})')
            initial = None
    if initial is not None:
        resolved = _attempt_with_encoder(initial, _encoder_debug_name(initial))
        if resolved is not None:
            return resolved

    current_name = getattr(config.encoder, 'name', None)
    for name in _encoder_candidate_names(config, baseline):
        if current_name is not None and str(name).lower() == str(current_name).lower() and initial is not None:
            continue
        try:
            candidate_encoder = build_encoder(_copy_encoder_config(config, name=name))
        except Exception as exc:
            build_errors.append(f'{name}: build failed ({exc})')
            continue
        resolved = _attempt_with_encoder(candidate_encoder, str(name))
        if resolved is not None:
            config.encoder.name = str(name)
            return resolved

    tried_text = ', '.join(tried_dims) if tried_dims else 'none'
    detail = '; '.join(build_errors[:6])
    raise FIPERPseudoOnsetError(
        'Feature dimension mismatch while relabeling pseudo-onsets: '
        f'baseline expects {expected_dim} dims, but relabel episode features did not match. '
        f'Tried {tried_text}. {detail}'.strip()
    )


def _infer_pseudo_onset_for_episode_internal(
    episode: Episode,
    baseline: FIPERStyleNormalBaseline,
    *,
    config: AppConfig | None = None,
    encoder=None,
) -> tuple[FIPERPseudoOnsetEpisodeResult, Any]:
    config = config or AppConfig()
    features, uncertainty, phase_idx, resolved_encoder = _resolve_episode_features_for_baseline(
        episode,
        baseline,
        config=config,
        encoder=encoder,
    )
    raw_obs = _episode_obs_scores(features, phase_idx, baseline)
    agg_obs = _rolling_sum(raw_obs, baseline.window)
    agg_uncertainty = _rolling_sum(uncertainty, baseline.window)
    obs_thresholds = np.asarray([
        baseline.obs_thresholds[int(phase)] if baseline.phase_feature_count[int(phase)] >= baseline.min_phase_count else baseline.global_obs_threshold
        for phase in phase_idx
    ], dtype=np.float32)
    uncertainty_thresholds = np.asarray([
        baseline.uncertainty_thresholds[int(phase)] if baseline.phase_feature_count[int(phase)] >= baseline.min_phase_count else baseline.global_uncertainty_threshold
        for phase in phase_idx
    ], dtype=np.float32)
    trigger_mask = (agg_obs >= obs_thresholds) & (agg_uncertainty >= uncertainty_thresholds)
    joint = np.minimum(
        agg_obs / np.maximum(obs_thresholds, _EPS),
        agg_uncertainty / np.maximum(uncertainty_thresholds, _EPS),
    )
    peak_joint = float(np.max(joint) if joint.size else 0.0)
    peak_index = int(np.argmax(joint)) if joint.size else None
    should_assign_failure = str(episode.outcome or '').lower() != 'success' or episode.failure_onset is not None
    if trigger_mask.any() and should_assign_failure:
        onset = int(np.argmax(trigger_mask))
        reason = 'dual_threshold'
        trigger_index = onset
    elif should_assign_failure:
        low_confidence = peak_joint <= 0.10 or (peak_index == 0 and peak_joint < 0.50)
        if low_confidence and episode.failure_onset is not None:
            onset = int(episode.failure_onset)
            reason = 'kept_original_low_confidence'
            trigger_index = None
        else:
            onset = peak_index
            reason = 'failure_peak_fallback'
            trigger_index = None
    else:
        onset = None
        reason = 'success_alarm_ignored' if trigger_mask.any() else 'success_no_alarm'
        trigger_index = int(np.argmax(trigger_mask)) if trigger_mask.any() else None
    return FIPERPseudoOnsetEpisodeResult(
        pseudo_failure_onset=onset,
        reason=reason,
        confidence=peak_joint,
        trigger_index=trigger_index,
        obs_scores=agg_obs,
        uncertainty_scores=agg_uncertainty,
        obs_thresholds=obs_thresholds,
        uncertainty_thresholds=uncertainty_thresholds,
        trigger_mask=trigger_mask,
    ), resolved_encoder


def infer_pseudo_onset_for_episode(
    episode: Episode,
    baseline: FIPERStyleNormalBaseline,
    *,
    config: AppConfig | None = None,
    encoder=None,
) -> FIPERPseudoOnsetEpisodeResult:
    result, _ = _infer_pseudo_onset_for_episode_internal(
        episode,
        baseline,
        config=config,
        encoder=encoder,
    )
    return result


def relabel_fino_manifest_with_pseudo_onsets(
    manifest_path: str | Path,
    baseline_path: str | Path,
    output_path: str | Path,
    *,
    image_size: int = 96,
    replace_failure_onset: bool = True,
    config: AppConfig | None = None,
    show_progress: bool = True,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 32,
    resume: bool = True,
) -> PseudoOnsetManifestResult:
    config = config or AppConfig()
    baseline = FIPERStyleNormalBaseline.load(baseline_path)
    if str(getattr(config.encoder, 'name', 'fallback')).strip().lower() in {'', 'fallback'} and baseline.encoder:
        config.encoder.name = baseline.encoder
    encoder = None
    entries = _load_manifest(manifest_path)
    output_path = Path(output_path)
    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else output_path.parent / f'{output_path.stem}.pseudo_onset_ckpt.json'

    rows: list[dict[str, Any]] = []
    start_idx = 0
    if resume and ckpt_path.exists() and output_path.exists():
        state = json.loads(ckpt_path.read_text(encoding='utf-8'))
        start_idx = int(state.get('next_index', 0))
        if output_path.suffix.lower() == '.jsonl':
            with output_path.open('r', encoding='utf-8') as f:
                rows = [json.loads(line) for line in f if line.strip()]
        else:
            rows = json.loads(output_path.read_text(encoding='utf-8'))

    total = len(entries)
    failure_episodes = 0
    success_episodes = 0
    pseudo_labeled_failures = 0
    replaced_failure_onsets = 0
    preserved_original_failure_onsets = 0
    for row in rows:
        outcome = str(row.get('outcome', '')).lower()
        if outcome == 'success':
            success_episodes += 1
        else:
            failure_episodes += 1
            if row.get('pseudo_failure_onset') not in (None, '', -1):
                pseudo_labeled_failures += 1
        if row.get('original_failure_onset') not in (None, '', -1):
            preserved_original_failure_onsets += 1
        if row.get('failure_onset') == row.get('pseudo_failure_onset') and row.get('pseudo_failure_onset') not in (None, '', -1):
            replaced_failure_onsets += 1

    progress_label = 'label-droid-failure-pseudo-onset' if 'droid' in str(manifest_path).lower() else 'label-fino-pseudo-onset'
    progress = ETATracker(label=progress_label, total=total, unit='episodes', print_every=1, initial_current=start_idx) if show_progress else None
    if progress is not None and start_idx > 0:
        progress.update(start_idx, extra=f'resume_from={start_idx}', force=True)

    for idx in range(start_idx, total):
        entry = dict(entries[idx])
        episode = _runtime_episode_from_manifest_entry(entry, image_size=image_size)
        result, encoder = _infer_pseudo_onset_for_episode_internal(episode, baseline, config=config, encoder=encoder)
        outcome = str(entry.get('outcome', 'failure' if entry.get('failure_onset') not in (None, '', -1) else 'success')).lower()
        if outcome == 'success':
            success_episodes += 1
        else:
            failure_episodes += 1
            if result.pseudo_failure_onset is not None:
                pseudo_labeled_failures += 1
        if entry.get('failure_onset') not in (None, '', -1):
            if entry.get('original_failure_onset') in (None, '', -1):
                entry['original_failure_onset'] = entry.get('failure_onset')
                preserved_original_failure_onsets += 1
        entry['pseudo_failure_onset'] = result.pseudo_failure_onset
        entry['pseudo_onset_reason'] = result.reason
        entry['pseudo_onset_confidence'] = round(float(result.confidence), 6)
        if result.trigger_index is not None:
            entry['pseudo_onset_trigger_index'] = int(result.trigger_index)
        if replace_failure_onset:
            if result.pseudo_failure_onset is not None:
                entry['failure_onset'] = int(result.pseudo_failure_onset)
                replaced_failure_onsets += 1
            elif outcome == 'success':
                entry['failure_onset'] = None
        rows.append(entry)
        if progress is not None:
            progress.update(idx + 1, extra=f"episode={entry.get('episode_id', idx)} reason={result.reason}")
        if checkpoint_every > 0 and ((idx + 1) % checkpoint_every == 0):
            _write_manifest_rows(output_path, rows)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_path.write_text(json.dumps({'next_index': idx + 1, 'completed': False, 'manifest_path': str(manifest_path)}, indent=2), encoding='utf-8')

    if progress is not None:
        progress.done(current=total, extra='pseudo-onset relabel finished')
    _write_manifest_rows(output_path, rows)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text(json.dumps({'next_index': total, 'completed': True, 'manifest_path': str(manifest_path)}, indent=2), encoding='utf-8')
    return PseudoOnsetManifestResult(
        baseline_path=str(Path(baseline_path)),
        output_path=str(output_path),
        total_episodes=total,
        failure_episodes=failure_episodes,
        pseudo_labeled_failures=pseudo_labeled_failures,
        success_episodes=success_episodes,
        replaced_failure_onsets=replaced_failure_onsets,
        preserved_original_failure_onsets=preserved_original_failure_onsets,
        checkpoint_path=str(ckpt_path),
    )



def rebuild_fino_with_pseudo_onset(
    droid_shard_dir: str | Path,
    fino_manifest_path: str | Path,
    base_bundle: str | Path,
    *,
    baseline_output_path: str | Path,
    pseudo_manifest_output_path: str | Path,
    converted_output_dir: str | Path,
    output_bundle_path: str | Path,
    config: AppConfig | None = None,
    epochs: int = 3,
    feature_source: str = 'visual',
    window: int = 3,
    phase_bins: int = 10,
    quantile: float = 0.97,
    min_phase_count: int = 8,
    image_size: int = 96,
    update_scaler: bool = False,
    show_progress: bool = True,
    fit_max_episodes: int | None = None,
    replace_failure_onset: bool = True,
    prefer_pseudo_onset: bool = True,
    pseudo_checkpoint_path: str | Path | None = None,
    pseudo_checkpoint_every: int = 32,
    pseudo_resume: bool = True,
    fino_checkpoint_path: str | Path | None = None,
    fino_checkpoint_every_shards: int = 1,
    fino_resume: bool = True,
) -> PseudoOnsetRebuildResult:
    config = config or AppConfig()
    baseline = fit_droid_success_baseline(
        droid_shard_dir,
        output_path=baseline_output_path,
        config=config,
        feature_source=feature_source,
        window=window,
        phase_bins=phase_bins,
        quantile=quantile,
        min_phase_count=min_phase_count,
        max_episodes=fit_max_episodes,
        show_progress=show_progress,
    )
    relabel_fino_manifest_with_pseudo_onsets(
        fino_manifest_path,
        baseline_output_path,
        pseudo_manifest_output_path,
        image_size=image_size,
        replace_failure_onset=replace_failure_onset,
        config=config,
        show_progress=show_progress,
        checkpoint_path=pseudo_checkpoint_path,
        checkpoint_every=pseudo_checkpoint_every,
        resume=pseudo_resume,
    )
    convert_failure_manifest_to_shards(
        pseudo_manifest_output_path,
        converted_output_dir,
        source_label=str(pseudo_manifest_output_path),
        source_mode='fino_manifest_pseudo_onset',
        image_size=image_size,
        show_progress=show_progress,
        prefer_pseudo_onset=prefer_pseudo_onset,
    )
    result = fine_tune_bundle_on_failure_shards(
        base_bundle,
        converted_output_dir,
        output_path=output_bundle_path,
        config=config,
        epochs=epochs,
        update_scaler=update_scaler,
        show_progress=show_progress,
        checkpoint_path=fino_checkpoint_path,
        checkpoint_every_shards=fino_checkpoint_every_shards,
        resume=fino_resume,
    )
    return PseudoOnsetRebuildResult(
        baseline_path=str(Path(baseline_output_path)),
        pseudo_manifest_path=str(Path(pseudo_manifest_output_path)),
        converted_root=str(Path(converted_output_dir)),
        output_bundle=str(Path(output_bundle_path)),
        metrics=result.metrics,
        strategies=result.strategies,
    )
