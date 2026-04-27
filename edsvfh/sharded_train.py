from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy

import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .calibration import PlattCalibrator
from .checkpointing import atomic_write_pickle, load_pickle
from .progress import ETATracker
from .config import AppConfig
from .models import VerifierBundle
from .public_data import list_hdf5_shards, load_robomimic_hdf5
from .train_public import build_feature_dataset


@dataclass
class ShardedTrainResult:
    bundle: VerifierBundle
    metrics: dict
    train_shards: list[str]
    calib_shards: list[str]
    eval_shards: list[str]
    checkpoint_path: str | None = None


class _ShardAccumulator:
    def __init__(self) -> None:
        self.items: list[np.ndarray] = []

    def add(self, arr: np.ndarray) -> None:
        if arr.size:
            self.items.append(np.asarray(arr))

    def concat(self) -> np.ndarray:
        if not self.items:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(self.items, axis=0)


def _collect_split_shards(shard_dir: str | Path, split: str) -> list[Path]:
    return list_hdf5_shards(shard_dir, split=split)


def _build_feature_dataset_from_shard(path: str | Path, config: AppConfig):
    episodes = load_robomimic_hdf5(path, config=config.dataset)
    return build_feature_dataset(episodes, config)


def _predict_binary_prob(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(x)
        prob = np.asarray(prob, dtype=np.float64)
        return prob[:, 1] if prob.ndim == 2 and prob.shape[1] > 1 else prob.reshape(-1)
    if hasattr(model, 'decision_function'):
        z = np.asarray(model.decision_function(x), dtype=np.float64)
        z = np.clip(z, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z))
    pred = np.asarray(model.predict(x), dtype=np.float64)
    return np.clip(pred, 0.0, 1.0)


def _default_checkpoint_path(shard_dir: str | Path, output_path: str | Path | None) -> Path:
    if output_path is not None:
        out = Path(output_path)
        return out.parent / f'{out.stem}.train_ckpt.pkl'
    return Path(shard_dir) / '.train_ckpt.pkl'


def _save_checkpoint(path: Path, payload: dict) -> None:
    atomic_write_pickle(path, payload)


def train_bundle_from_shards(
    shard_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    config: AppConfig | None = None,
    epochs: int = 1,
    delete_consumed_train_shards: bool = False,
    show_progress: bool = True,
    checkpoint_path: str | Path | None = None,
    checkpoint_every_shards: int = 1,
    resume: bool = True,
) -> ShardedTrainResult:
    config = config or AppConfig()
    train_shards = _collect_split_shards(shard_dir, 'train')
    calib_shards = _collect_split_shards(shard_dir, 'calib')
    eval_shards = _collect_split_shards(shard_dir, 'eval')
    if not train_shards:
        raise FileNotFoundError(f'No train shards found under {Path(shard_dir) / "train"}')
    if delete_consumed_train_shards and epochs != 1:
        raise ValueError('delete_consumed_train_shards=True only supports epochs=1 in this reference implementation.')
    if delete_consumed_train_shards and resume:
        raise ValueError('resume=True is not supported together with delete_consumed_train_shards=True.')

    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else _default_checkpoint_path(shard_dir, output_path)

    scaler: StandardScaler | None = None
    subgoal_model: SGDClassifier | None = None
    completion_model: SGDRegressor | None = None
    done_model: SGDClassifier | None = None
    horizon_models: list[SGDClassifier | None] = [None for _ in config.training.horizons]
    horizon_seen = [set() for _ in config.training.horizons]
    input_dim: int | None = None
    first_multiclass = True
    first_done = True
    first_horizon = [True for _ in config.training.horizons]
    next_epoch = 0
    next_shard_idx = 0
    train_processed = 0

    if resume and ckpt_path.exists():
        state = load_pickle(ckpt_path)
        if state.get('completed'):
            if output_path is not None and Path(output_path).exists():
                bundle = VerifierBundle.load(output_path)
                metrics = evaluate_bundle_on_shards(
                    bundle,
                    eval_shards if eval_shards else calib_shards if calib_shards else train_shards,
                    config,
                    show_progress=show_progress,
                    progress_label='train-sharded:eval',
                )
                return ShardedTrainResult(
                    bundle=bundle,
                    metrics=metrics,
                    train_shards=[str(p) for p in train_shards],
                    calib_shards=[str(p) for p in calib_shards],
                    eval_shards=[str(p) for p in eval_shards],
                    checkpoint_path=str(ckpt_path),
                )
        scaler = state['scaler']
        subgoal_model = state['subgoal_model']
        completion_model = state['completion_model']
        done_model = state['done_model']
        horizon_models = state['horizon_models']
        horizon_seen = state['horizon_seen']
        input_dim = state['input_dim']
        first_multiclass = state['first_multiclass']
        first_done = state['first_done']
        first_horizon = state['first_horizon']
        next_epoch = state['next_epoch']
        next_shard_idx = state['next_shard_idx']
        train_processed = state['train_processed']

    train_total = max(1, epochs * len(train_shards))
    train_progress = ETATracker(
        label='train-sharded:train',
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
            X = dataset.X.astype(np.float32)
            y_subgoal = dataset.y_subgoal.astype(np.int64)
            y_completion = dataset.y_completion.astype(np.float32)
            y_done = dataset.y_done.astype(int)
            y_h = dataset.y_h.astype(int)
            if input_dim is None:
                input_dim = X.shape[1]
                scaler = StandardScaler()
                subgoal_model = SGDClassifier(loss='log_loss', random_state=config.training.random_seed, average=True)
                completion_model = SGDRegressor(loss='huber', random_state=config.training.random_seed)
                done_model = SGDClassifier(loss='log_loss', random_state=config.training.random_seed + 100, average=True)
                horizon_models = [
                    SGDClassifier(loss='log_loss', random_state=config.training.random_seed + 200 + i, average=True)
                    for i, _ in enumerate(config.training.horizons)
                ]
            assert scaler is not None and subgoal_model is not None and completion_model is not None and done_model is not None
            scaler.partial_fit(X)
            Xs = scaler.transform(X)

            subgoal_classes = np.arange(config.training.num_subgoals, dtype=np.int64)
            if first_multiclass:
                subgoal_model.partial_fit(Xs, y_subgoal, classes=subgoal_classes)
                first_multiclass = False
            else:
                subgoal_model.partial_fit(Xs, y_subgoal)

            completion_model.partial_fit(Xs, y_completion)

            bin_classes = np.array([0, 1], dtype=np.int64)
            if first_done:
                done_model.partial_fit(Xs, y_done, classes=bin_classes)
                first_done = False
            else:
                done_model.partial_fit(Xs, y_done)

            for k, _ in enumerate(config.training.horizons):
                labels = y_h[:, k]
                horizon_seen[k].update(np.unique(labels).tolist())
                if horizon_models[k] is None:
                    continue
                if first_horizon[k]:
                    horizon_models[k].partial_fit(Xs, labels, classes=bin_classes)
                    first_horizon[k] = False
                else:
                    horizon_models[k].partial_fit(Xs, labels)

            train_processed += 1
            checkpoint_counter += 1
            if train_progress is not None:
                train_progress.update(
                    train_processed,
                    extra=f"epoch={epoch + 1}/{epochs} shard={Path(shard_path).name} samples={X.shape[0]}",
                )
            future_epoch = epoch
            future_shard = shard_idx + 1
            if future_shard >= len(train_shards):
                future_epoch = epoch + 1
                future_shard = 0
            if checkpoint_every_shards > 0 and checkpoint_counter % checkpoint_every_shards == 0:
                _save_checkpoint(ckpt_path, {
                    'completed': False,
                    'scaler': scaler,
                    'subgoal_model': subgoal_model,
                    'completion_model': completion_model,
                    'done_model': done_model,
                    'horizon_models': horizon_models,
                    'horizon_seen': horizon_seen,
                    'input_dim': input_dim,
                    'first_multiclass': first_multiclass,
                    'first_done': first_done,
                    'first_horizon': first_horizon,
                    'next_epoch': future_epoch,
                    'next_shard_idx': future_shard,
                    'train_processed': train_processed,
                    'epochs': epochs,
                })
            if delete_consumed_train_shards:
                Path(shard_path).unlink(missing_ok=True)

    if train_progress is not None:
        train_progress.done(current=train_processed, extra='feature accumulation and partial-fit finished')

    assert input_dim is not None and scaler is not None and subgoal_model is not None and completion_model is not None and done_model is not None

    effective_horizon_models: list[object | None] = []
    for k, model in enumerate(horizon_models):
        if horizon_seen[k] == {0}:
            effective_horizon_models.append(None)
        elif horizon_seen[k] == {1}:
            effective_horizon_models.append(None)
        else:
            effective_horizon_models.append(model)

    calib_source = calib_shards if calib_shards else eval_shards
    horizon_calibrators: list[PlattCalibrator] = []
    calib_total = len(calib_source) * max(1, len(config.training.horizons))
    calib_progress = ETATracker(
        label='train-sharded:calib',
        total=calib_total,
        unit='passes',
        print_every=1,
    ) if show_progress and calib_total > 0 else None
    calib_processed = 0
    for k, horizon in enumerate(config.training.horizons):
        if effective_horizon_models[k] is None:
            constant = 0.0 if horizon_seen[k] != {1} else 1.0
            horizon_calibrators.append(PlattCalibrator(model=None, constant=constant))
            continue
        prob_acc = _ShardAccumulator()
        label_acc = _ShardAccumulator()
        for shard_path in calib_source:
            dataset = _build_feature_dataset_from_shard(shard_path, config)
            Xs = scaler.transform(dataset.X.astype(np.float32))
            prob_acc.add(_predict_binary_prob(effective_horizon_models[k], Xs))
            label_acc.add(dataset.y_h[:, k].astype(int))
            calib_processed += 1
            if calib_progress is not None:
                calib_progress.update(calib_processed, extra=f"horizon={horizon} shard={Path(shard_path).name}")
        base_prob = prob_acc.concat()
        labels = label_acc.concat().astype(int)
        if base_prob.size == 0:
            horizon_calibrators.append(PlattCalibrator(model=None, constant=0.5))
        else:
            horizon_calibrators.append(PlattCalibrator.fit(base_prob, labels, seed=config.training.random_seed + k))

    if calib_progress is not None:
        calib_progress.done(current=calib_processed, extra='calibration finished')

    bundle = VerifierBundle(
        subgoal_model=subgoal_model,
        completion_model=completion_model,
        done_model=done_model,
        horizon_models=effective_horizon_models,
        horizon_calibrators=horizon_calibrators,
        horizons=config.training.horizons,
        input_dim=input_dim,
        num_subgoals=config.training.num_subgoals,
        metadata={
            'encoder': config.encoder.name,
            'seed': config.training.random_seed,
            'training_mode': 'sharded_incremental',
            'epochs': epochs,
            'num_train_shards': len(train_shards),
        },
        feature_scaler=scaler,
    )

    metrics = evaluate_bundle_on_shards(
        bundle,
        eval_shards if eval_shards else calib_shards if calib_shards else train_shards,
        config,
        show_progress=show_progress,
        progress_label='train-sharded:eval',
    )
    if output_path is not None:
        bundle.save(output_path)
    _save_checkpoint(ckpt_path, {
        'completed': True,
        'output_path': str(output_path) if output_path is not None else None,
        'epochs': epochs,
    })
    return ShardedTrainResult(
        bundle=bundle,
        metrics=metrics,
        train_shards=[str(p) for p in train_shards],
        calib_shards=[str(p) for p in calib_shards],
        eval_shards=[str(p) for p in eval_shards],
        checkpoint_path=str(ckpt_path),
    )


def evaluate_bundle_on_shards(
    bundle: VerifierBundle,
    shard_paths: list[Path] | list[str],
    config: AppConfig,
    *,
    show_progress: bool = False,
    progress_label: str = 'evaluate',
) -> dict:
    config = copy.deepcopy(config)
    config.training.horizons = tuple(bundle.horizons)

    sg_true = _ShardAccumulator()
    sg_pred = _ShardAccumulator()
    comp_true = _ShardAccumulator()
    comp_pred = _ShardAccumulator()
    done_true = _ShardAccumulator()
    done_prob = _ShardAccumulator()
    h_true = [_ShardAccumulator() for _ in bundle.horizons]
    h_prob = [_ShardAccumulator() for _ in bundle.horizons]

    eval_progress = ETATracker(
        label=progress_label,
        total=len(shard_paths),
        unit='shards',
        print_every=1,
    ) if show_progress and shard_paths else None

    processed = 0
    for shard_path in shard_paths:
        dataset = _build_feature_dataset_from_shard(shard_path, config)
        pred_sg, pred_comp, pred_done, pred_h = bundle.predict(dataset.X.astype(np.float32))
        sg_true.add(dataset.y_subgoal.astype(np.int64))
        sg_pred.add(pred_sg.astype(np.int64))
        comp_true.add(dataset.y_completion.astype(np.float32))
        comp_pred.add(pred_comp.astype(np.float32))
        done_true.add(dataset.y_done.astype(np.int64))
        done_prob.add(pred_done.astype(np.float32))
        for k, _ in enumerate(bundle.horizons):
            h_true[k].add(dataset.y_h[:, k].astype(np.int64))
            h_prob[k].add(pred_h[:, k].astype(np.float32))
        processed += 1
        if eval_progress is not None:
            eval_progress.update(processed, extra=f"shard={Path(shard_path).name} samples={dataset.X.shape[0]}")

    if eval_progress is not None:
        eval_progress.done(current=processed, extra='evaluation finished')

    y_sg = sg_true.concat().astype(np.int64)
    p_sg = sg_pred.concat().astype(np.int64)
    y_comp = comp_true.concat().astype(np.float32)
    p_comp = comp_pred.concat().astype(np.float32)
    y_done = done_true.concat().astype(np.int64)
    p_done = done_prob.concat().astype(np.float32)

    metrics: dict[str, float] = {
        'subgoal_accuracy': float(accuracy_score(y_sg, p_sg)) if y_sg.size else float('nan'),
        'subgoal_macro_f1': float(f1_score(y_sg, p_sg, average='macro')) if y_sg.size else float('nan'),
        'completion_mae': float(mean_absolute_error(y_comp, p_comp)) if y_comp.size else float('nan'),
        'done_accuracy': float(accuracy_score(y_done, (p_done >= 0.5).astype(int))) if y_done.size else float('nan'),
    }
    aucs = []
    for k, horizon in enumerate(bundle.horizons):
        y = h_true[k].concat().astype(np.int64)
        p = h_prob[k].concat().astype(np.float32)
        if y.size == 0 or len(np.unique(y)) < 2:
            metrics[f'horizon_{horizon}_auc'] = float('nan')
            continue
        auc = float(roc_auc_score(y, p))
        metrics[f'horizon_{horizon}_auc'] = auc
        aucs.append(auc)
    metrics['mean_horizon_auc'] = float(np.nanmean(aucs)) if aucs else float('nan')
    return metrics
