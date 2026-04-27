from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from .calibration import PlattCalibrator
from .checkpointing import atomic_write_pickle, load_pickle
from .progress import ETATracker
from .config import AppConfig
from .models import VerifierBundle, adapt_feature_dim
from .public_data import list_hdf5_shards, load_robomimic_hdf5
from .sharded_train import evaluate_bundle_on_shards
from .train_public import build_feature_dataset


@dataclass
class FailureFineTuneResult:
    bundle: VerifierBundle
    metrics: dict
    train_shards: list[str]
    calib_shards: list[str]
    eval_shards: list[str]
    strategies: list[str]
    checkpoint_path: str | None = None


class FailureFineTuneError(RuntimeError):
    pass


class _Accumulator:
    def __init__(self) -> None:
        self.items: list[np.ndarray] = []

    def add(self, arr: np.ndarray) -> None:
        arr = np.asarray(arr)
        if arr.size:
            self.items.append(arr)

    def concat(self) -> np.ndarray:
        if not self.items:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(self.items, axis=0)


def _build_feature_dataset_from_shard(path: str | Path, config: AppConfig):
    episodes = load_robomimic_hdf5(path, config=config.dataset)
    return build_feature_dataset(episodes, config)


def _ensure_scaler(bundle: VerifierBundle, train_shards: list[Path], config: AppConfig, update_scaler: bool) -> StandardScaler:
    if bundle.feature_scaler is None:
        scaler = StandardScaler()
        for shard_path in train_shards:
            dataset = _build_feature_dataset_from_shard(shard_path, config)
            scaler.partial_fit(adapt_feature_dim(dataset.X.astype(np.float32), bundle.input_dim))
        return scaler

    scaler = copy.deepcopy(bundle.feature_scaler) if update_scaler else bundle.feature_scaler
    if update_scaler and hasattr(scaler, 'partial_fit'):
        for shard_path in train_shards:
            dataset = _build_feature_dataset_from_shard(shard_path, config)
            scaler.partial_fit(adapt_feature_dim(dataset.X.astype(np.float32), bundle.input_dim))
    return scaler


def _default_checkpoint_path(base_bundle: str | Path, output_path: str | Path | None, shard_dir: str | Path) -> Path:
    if output_path is not None:
        out = Path(output_path)
        return out.parent / f'{out.stem}.fino_ckpt.pkl'
    base = Path(base_bundle) if isinstance(base_bundle, (str, Path)) else Path(shard_dir)
    return base.parent / f'{base.stem}.fino_ckpt.pkl'


def fine_tune_bundle_on_failure_shards(
    base_bundle: str | Path | VerifierBundle,
    shard_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    config: AppConfig | None = None,
    epochs: int = 3,
    update_scaler: bool = False,
    show_progress: bool = True,
    checkpoint_path: str | Path | None = None,
    checkpoint_every_shards: int = 1,
    resume: bool = True,
    freeze_existing_horizons: bool = False,
) -> FailureFineTuneResult:
    config = config or AppConfig()
    bundle = VerifierBundle.load(base_bundle) if isinstance(base_bundle, (str, Path)) else base_bundle
    old_horizons = tuple(int(h) for h in bundle.horizons)
    requested_horizons = tuple(int(h) for h in config.training.horizons)

    # v29: when expanding horizons, keep overlapping heads by horizon value instead
    # of reinitializing the whole head bank. This preserves the already validated
    # 1/3/5 heads and trains only newly requested horizons such as 10/15.
    inherited_horizon_flags = [h in old_horizons for h in requested_horizons]
    if requested_horizons != old_horizons:
        old_index = {h: i for i, h in enumerate(old_horizons)}
        old_models = list(bundle.horizon_models)
        old_calibrators = list(bundle.horizon_calibrators)
        bundle = copy.deepcopy(bundle)
        new_models: list[object | None] = []
        new_calibrators: list[PlattCalibrator] = []
        for h in requested_horizons:
            if h in old_index:
                idx = old_index[h]
                new_models.append(copy.deepcopy(old_models[idx]) if idx < len(old_models) else None)
                if idx < len(old_calibrators):
                    new_calibrators.append(copy.deepcopy(old_calibrators[idx]))
                else:
                    new_calibrators.append(PlattCalibrator(model=None, constant=0.5))
            else:
                new_models.append(None)
                new_calibrators.append(PlattCalibrator(model=None, constant=0.5))
        bundle.horizons = requested_horizons
        bundle.horizon_models = new_models
        bundle.horizon_calibrators = new_calibrators
        bundle.metadata = {
            **bundle.metadata,
            'failure_finetune_horizon_override_from': list(old_horizons),
            'failure_finetune_horizon_override_to': list(requested_horizons),
            'failure_finetune_horizon_overlap_warm_start': True,
        }
    freeze_horizon_flags = [bool(freeze_existing_horizons and inherited) for inherited in inherited_horizon_flags]
    if update_scaler and any(freeze_horizon_flags):
        raise FailureFineTuneError(
            'update_scaler=True is incompatible with freeze_existing_horizons=True because '
            'changing the feature scaler invalidates frozen horizon heads. Set update_scaler=False.'
        )
    train_shards = list_hdf5_shards(shard_dir, split='train')
    calib_shards = list_hdf5_shards(shard_dir, split='calib')
    eval_shards = list_hdf5_shards(shard_dir, split='eval')
    if not train_shards:
        raise FailureFineTuneError(f'No train shards found under {Path(shard_dir) / "train"}')

    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else _default_checkpoint_path(base_bundle if isinstance(base_bundle, (str, Path)) else shard_dir, output_path, shard_dir)

    scaler = _ensure_scaler(bundle, train_shards, config, update_scaler=update_scaler)
    bundle.feature_scaler = scaler

    bin_classes = np.array([0, 1], dtype=np.int64)
    strategies: list[str] = []
    horizon_models: list[object | None] = list(bundle.horizon_models)
    frozen_calibrators: list[PlattCalibrator] = [copy.deepcopy(c) for c in bundle.horizon_calibrators]
    first_flags: list[bool] = []
    train_horizon_flags: list[bool] = [not frozen for frozen in freeze_horizon_flags]
    next_epoch = 0
    next_shard_idx = 0
    train_processed = 0

    if resume and ckpt_path.exists():
        state = load_pickle(ckpt_path)
        if isinstance(state, dict) and state.get('completed'):
            if output_path is not None and Path(output_path).exists():
                bundle = VerifierBundle.load(output_path)
                metrics = evaluate_bundle_on_shards(
                    bundle,
                    eval_shards if eval_shards else calib_shards if calib_shards else train_shards,
                    config,
                    show_progress=show_progress,
                    progress_label='fine-tune-fino:eval',
                )
                return FailureFineTuneResult(
                    bundle=bundle,
                    metrics=metrics,
                    train_shards=[str(p) for p in train_shards],
                    calib_shards=[str(p) for p in calib_shards],
                    eval_shards=[str(p) for p in eval_shards],
                    strategies=state.get('strategies', []),
                    checkpoint_path=str(ckpt_path),
                )
            # Completed checkpoint without output bundle present: restart cleanly.
            state = None

        if isinstance(state, dict) and {'bundle', 'scaler', 'horizon_models', 'first_flags', 'strategies', 'next_epoch', 'next_shard_idx', 'train_processed'} <= set(state.keys()) and tuple(state.get('horizons', bundle.horizons)) == tuple(bundle.horizons):
            bundle = state['bundle']
            scaler = state['scaler']
            horizon_models = state['horizon_models']
            frozen_calibrators = state.get('frozen_calibrators', frozen_calibrators)
            first_flags = state['first_flags']
            strategies = state['strategies']
            train_horizon_flags = state.get('train_horizon_flags', train_horizon_flags)
            next_epoch = state['next_epoch']
            next_shard_idx = state['next_shard_idx']
            train_processed = state['train_processed']
        else:
            # Missing or legacy checkpoint schema: ignore and restart from scratch.
            next_epoch = 0
            next_shard_idx = 0
            train_processed = 0
            strategies = []
            horizon_models = list(bundle.horizon_models)
            first_flags = []
            if ckpt_path.exists():
                ckpt_path.unlink(missing_ok=True)
            for k, model in enumerate(horizon_models):
                if not train_horizon_flags[k]:
                    strategies.append('kept_existing_frozen')
                    first_flags.append(False)
                elif model is not None and hasattr(model, 'partial_fit'):
                    strategies.append('continued_partial_fit')
                    first_flags.append(False)
                else:
                    horizon_models[k] = SGDClassifier(loss='log_loss', random_state=config.training.random_seed + 500 + k, average=True)
                    strategies.append('reinitialized_sgd')
                    first_flags.append(True)
    else:
        for k, model in enumerate(horizon_models):
            if not train_horizon_flags[k]:
                strategies.append('kept_existing_frozen')
                first_flags.append(False)
            elif model is not None and hasattr(model, 'partial_fit'):
                strategies.append('continued_partial_fit')
                first_flags.append(False)
            else:
                horizon_models[k] = SGDClassifier(loss='log_loss', random_state=config.training.random_seed + 500 + k, average=True)
                strategies.append('reinitialized_sgd')
                first_flags.append(True)

    train_total = max(1, epochs * len(train_shards))
    train_progress = ETATracker(
        label='fine-tune-fino:train',
        total=train_total,
        unit='shards',
        print_every=1,
        initial_current=train_processed,
    ) if show_progress else None
    if train_progress is not None and train_processed > 0:
        train_progress.update(train_processed, extra=f'resume epoch={next_epoch + 1}/{epochs} shard_idx={next_shard_idx}', force=True)

    checkpoint_counter = 0
    for epoch in range(next_epoch, epochs):
        shard_start = next_shard_idx if epoch == next_epoch else 0
        for shard_idx in range(shard_start, len(train_shards)):
            shard_path = train_shards[shard_idx]
            dataset = _build_feature_dataset_from_shard(shard_path, config)
            Xs = scaler.transform(adapt_feature_dim(dataset.X.astype(np.float32), bundle.input_dim))
            y_h = dataset.y_h.astype(int)
            for k, _ in enumerate(bundle.horizons):
                if not train_horizon_flags[k]:
                    continue
                labels = y_h[:, k]
                model = horizon_models[k]
                if model is None:
                    continue
                if first_flags[k]:
                    model.partial_fit(Xs, labels, classes=bin_classes)
                    first_flags[k] = False
                else:
                    model.partial_fit(Xs, labels)
            train_processed += 1
            checkpoint_counter += 1
            if train_progress is not None:
                train_progress.update(train_processed, extra=f"epoch={epoch + 1}/{epochs} shard={Path(shard_path).name} samples={Xs.shape[0]}")
            future_epoch = epoch
            future_shard = shard_idx + 1
            if future_shard >= len(train_shards):
                future_epoch = epoch + 1
                future_shard = 0
            if checkpoint_every_shards > 0 and checkpoint_counter % checkpoint_every_shards == 0:
                atomic_write_pickle(ckpt_path, {
                    'completed': False,
                    'bundle': bundle,
                    'scaler': scaler,
                    'horizon_models': horizon_models,
                    'first_flags': first_flags,
                    'strategies': strategies,
                    'train_horizon_flags': train_horizon_flags,
                    'frozen_calibrators': frozen_calibrators,
                    'next_epoch': future_epoch,
                    'next_shard_idx': future_shard,
                    'train_processed': train_processed,
                    'horizons': list(bundle.horizons),
                })

    if train_progress is not None:
        train_progress.done(current=train_processed, extra='failure-side horizon adaptation finished')

    horizon_calibrators: list[PlattCalibrator] = []
    calib_source = calib_shards if calib_shards else eval_shards if eval_shards else train_shards
    calib_total = len(calib_source) * max(1, sum(1 for flag in train_horizon_flags if flag))
    calib_progress = ETATracker(
        label='fine-tune-fino:calib',
        total=calib_total,
        unit='passes',
        print_every=1,
    ) if show_progress and calib_total > 0 else None
    calib_processed = 0
    for k, _ in enumerate(bundle.horizons):
        model = horizon_models[k]
        if not train_horizon_flags[k]:
            horizon_calibrators.append(copy.deepcopy(frozen_calibrators[k]))
            continue
        if model is None:
            horizon_calibrators.append(PlattCalibrator(model=None, constant=0.5))
            continue
        prob_acc = _Accumulator()
        label_acc = _Accumulator()
        for shard_path in calib_source:
            dataset = _build_feature_dataset_from_shard(shard_path, config)
            Xs = scaler.transform(adapt_feature_dim(dataset.X.astype(np.float32), bundle.input_dim))
            if hasattr(model, 'predict_proba'):
                prob = np.asarray(model.predict_proba(Xs), dtype=np.float64)
                prob = prob[:, 1] if prob.ndim == 2 and prob.shape[1] > 1 else prob.reshape(-1)
            elif hasattr(model, 'decision_function'):
                z = np.asarray(model.decision_function(Xs), dtype=np.float64)
                z = np.clip(z, -40.0, 40.0)
                prob = 1.0 / (1.0 + np.exp(-z))
            else:
                prob = np.asarray(model.predict(Xs), dtype=np.float64)
            prob_acc.add(prob)
            label_acc.add(dataset.y_h[:, k].astype(int))
            calib_processed += 1
            if calib_progress is not None:
                calib_progress.update(calib_processed, extra=f"horizon={bundle.horizons[k]} shard={Path(shard_path).name}")
        base_prob = prob_acc.concat().astype(np.float64)
        labels = label_acc.concat().astype(int)
        if base_prob.size == 0 or len(np.unique(labels)) < 2:
            constant = float(labels[0]) if labels.size else 0.5
            horizon_calibrators.append(PlattCalibrator(model=None, constant=constant))
        else:
            horizon_calibrators.append(PlattCalibrator.fit(base_prob, labels, seed=config.training.random_seed + 700 + k))

    if calib_progress is not None:
        calib_progress.done(current=calib_processed, extra='calibration finished')

    bundle.horizon_models = horizon_models
    bundle.horizon_calibrators = horizon_calibrators
    bundle.metadata = {
        **bundle.metadata,
        'failure_finetune_source': str(shard_dir),
        'failure_finetune_epochs': int(epochs),
        'failure_finetune_update_scaler': bool(update_scaler),
        'failure_finetune_horizons': list(bundle.horizons),
        'failure_finetune_strategies': strategies,
        'failure_finetune_freeze_existing_horizons': bool(freeze_existing_horizons),
        'failure_finetune_train_horizon_flags': list(train_horizon_flags),
    }

    metrics = evaluate_bundle_on_shards(
        bundle,
        eval_shards if eval_shards else calib_shards if calib_shards else train_shards,
        config,
        show_progress=show_progress,
        progress_label='fine-tune-fino:eval',
    )
    if output_path is not None:
        bundle.save(output_path)
    atomic_write_pickle(ckpt_path, {
        'completed': True,
        'strategies': strategies,
        'output_path': str(output_path) if output_path is not None else None,
        'horizons': list(bundle.horizons),
        'train_horizon_flags': train_horizon_flags,
    })
    return FailureFineTuneResult(
        bundle=bundle,
        metrics=metrics,
        train_shards=[str(p) for p in train_shards],
        calib_shards=[str(p) for p in calib_shards],
        eval_shards=[str(p) for p in eval_shards],
        strategies=strategies,
        checkpoint_path=str(ckpt_path),
    )
