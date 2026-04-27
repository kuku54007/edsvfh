from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score

from .calibration import PlattCalibrator
from .config import AppConfig
from .context import ContextBuilder
from .encoders import build_encoder, snapshot_from_precomputed
from .memory import EventMemory
from .models import VerifierBundle
from .pseudo_labels import infer_horizon_labels, infer_step_label
from .public_data import create_tiny_robomimic_fixture, load_robomimic_hdf5
from .types import Episode, EventPacket


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_subgoal_train: np.ndarray
    y_subgoal_val: np.ndarray
    y_subgoal_test: np.ndarray
    y_completion_train: np.ndarray
    y_completion_val: np.ndarray
    y_completion_test: np.ndarray
    y_done_train: np.ndarray
    y_done_val: np.ndarray
    y_done_test: np.ndarray
    y_h_train: np.ndarray
    y_h_val: np.ndarray
    y_h_test: np.ndarray
    episode_ids_test: list[str]


@dataclass
class FeatureDataset:
    X: np.ndarray
    y_subgoal: np.ndarray
    y_completion: np.ndarray
    y_done: np.ndarray
    y_h: np.ndarray
    episode_ids: list[str]


def build_feature_dataset(episodes: list[Episode], config: AppConfig) -> FeatureDataset:
    encoder = None
    X: list[np.ndarray] = []
    y_subgoal: list[int] = []
    y_completion: list[float] = []
    y_done: list[float] = []
    y_h: list[np.ndarray] = []
    episode_ids: list[str] = []

    for ep_idx, episode in enumerate(episodes):
        ctx = ContextBuilder(window=config.memory.window)
        memory = EventMemory(capacity=config.memory.capacity, num_subgoals=config.training.num_subgoals, num_horizons=len(config.training.horizons))
        for step_idx, step in enumerate(episode.steps):
            gt = infer_step_label(episode, step_idx)
            h = infer_horizon_labels(episode, step_idx, config.training.horizons)
            if step.observation.precomputed_vector is not None:
                snapshot = snapshot_from_precomputed(step.observation)
            else:
                if encoder is None:
                    encoder = build_encoder(config.encoder)
                snapshot = encoder.extract(step.observation)
            ctx.update(snapshot)
            context = ctx.build(step.observation.timestamp, snapshot, memory)
            X.append(context)
            y_subgoal.append(gt.subgoal)
            y_completion.append(gt.completion)
            y_done.append(float(gt.done))
            y_h.append(h)
            episode_ids.append(f'{ep_idx}:{step_idx}')
            if gt.event:
                memory.add(
                    EventPacket(
                        timestamp=step.observation.timestamp,
                        subgoal=gt.subgoal,
                        completion=gt.completion,
                        done=float(gt.done),
                        risk=h,
                        visual_embedding=snapshot.visual_embedding,
                    )
                )

    return FeatureDataset(
        X=np.asarray(X, dtype=np.float32),
        y_subgoal=np.asarray(y_subgoal, dtype=np.int64),
        y_completion=np.asarray(y_completion, dtype=np.float32),
        y_done=np.asarray(y_done, dtype=np.float32),
        y_h=np.asarray(y_h, dtype=np.float32),
        episode_ids=episode_ids,
    )


def split_by_episode(dataset: FeatureDataset, seed: int, valid_ratio: float, test_ratio: float) -> DatasetSplit:
    episode_roots = sorted({eid.split(':')[0] for eid in dataset.episode_ids})
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_roots)
    n = len(episode_roots)
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * valid_ratio)))
    n_train = max(1, n - n_val - n_test)
    train_roots = set(episode_roots[:n_train])
    val_roots = set(episode_roots[n_train:n_train + n_val])
    test_roots = set(episode_roots[n_train + n_val:])

    def idx_for(roots: set[str]) -> np.ndarray:
        return np.array([i for i, eid in enumerate(dataset.episode_ids) if eid.split(':')[0] in roots], dtype=np.int64)

    train_idx = idx_for(train_roots)
    val_idx = idx_for(val_roots)
    test_idx = idx_for(test_roots)
    return DatasetSplit(
        X_train=dataset.X[train_idx],
        X_val=dataset.X[val_idx],
        X_test=dataset.X[test_idx],
        y_subgoal_train=dataset.y_subgoal[train_idx],
        y_subgoal_val=dataset.y_subgoal[val_idx],
        y_subgoal_test=dataset.y_subgoal[test_idx],
        y_completion_train=dataset.y_completion[train_idx],
        y_completion_val=dataset.y_completion[val_idx],
        y_completion_test=dataset.y_completion[test_idx],
        y_done_train=dataset.y_done[train_idx],
        y_done_val=dataset.y_done[val_idx],
        y_done_test=dataset.y_done[test_idx],
        y_h_train=dataset.y_h[train_idx],
        y_h_val=dataset.y_h[val_idx],
        y_h_test=dataset.y_h[test_idx],
        episode_ids_test=[dataset.episode_ids[i] for i in test_idx],
    )


def train_bundle_from_episodes(episodes: list[Episode], output_path: str | Path | None = None, config: AppConfig | None = None) -> tuple[VerifierBundle, dict]:
    config = config or AppConfig()
    dataset = build_feature_dataset(episodes, config)
    split = split_by_episode(dataset, seed=config.training.random_seed, valid_ratio=config.training.valid_ratio, test_ratio=config.training.test_ratio)

    subgoal_model = RandomForestClassifier(
        class_weight='balanced_subsample',
        random_state=config.training.random_seed,
        n_estimators=config.training.n_estimators,
        max_depth=config.training.max_depth,
        n_jobs=1,
    )
    subgoal_model.fit(split.X_train, split.y_subgoal_train)

    completion_model = RandomForestRegressor(
        random_state=config.training.random_seed,
        n_estimators=config.training.n_estimators,
        max_depth=config.training.max_depth,
        n_jobs=1,
    )
    completion_model.fit(split.X_train, split.y_completion_train)

    done_model = RandomForestClassifier(
        class_weight='balanced_subsample',
        random_state=config.training.random_seed,
        n_estimators=config.training.n_estimators,
        max_depth=config.training.max_depth,
        n_jobs=1,
    )
    done_model.fit(split.X_train, split.y_done_train.astype(int))

    horizon_models: list[RandomForestClassifier | None] = []
    horizon_calibrators: list[PlattCalibrator] = []
    for k, horizon in enumerate(config.training.horizons):
        labels_train = split.y_h_train[:, k].astype(int)
        labels_val = split.y_h_val[:, k].astype(int)
        if len(np.unique(labels_train)) < 2:
            horizon_models.append(None)
            horizon_calibrators.append(PlattCalibrator(model=None, constant=float(labels_train[0])))
            continue
        model = RandomForestClassifier(
            class_weight='balanced_subsample',
            random_state=config.training.random_seed + k,
            n_estimators=config.training.n_estimators + 20,
            max_depth=config.training.max_depth,
            n_jobs=1,
        )
        model.fit(split.X_train, labels_train)
        base_prob_val = model.predict_proba(split.X_val)[:, 1]
        calibrator = PlattCalibrator.fit(base_prob_val, labels_val, seed=config.training.random_seed + k)
        horizon_models.append(model)
        horizon_calibrators.append(calibrator)

    bundle = VerifierBundle(
        subgoal_model=subgoal_model,
        completion_model=completion_model,
        done_model=done_model,
        horizon_models=horizon_models,
        horizon_calibrators=horizon_calibrators,
        horizons=config.training.horizons,
        input_dim=dataset.X.shape[1],
        num_subgoals=config.training.num_subgoals,
        metadata={'encoder': config.encoder.name, 'seed': config.training.random_seed, 'num_steps': int(len(dataset.X))},
    )
    metrics = evaluate_bundle(bundle, split)
    if output_path is not None:
        bundle.save(output_path)
    return bundle, metrics


def evaluate_bundle(bundle: VerifierBundle, split: DatasetSplit) -> dict:
    sg_pred, completion_pred, done_pred, horizon_pred = bundle.predict(split.X_test)
    metrics: dict[str, float] = {
        'subgoal_accuracy': float(accuracy_score(split.y_subgoal_test, sg_pred)),
        'subgoal_macro_f1': float(f1_score(split.y_subgoal_test, sg_pred, average='macro')),
        'completion_mae': float(mean_absolute_error(split.y_completion_test, completion_pred)),
        'done_accuracy': float(accuracy_score(split.y_done_test.astype(int), (done_pred >= 0.5).astype(int))),
    }
    aucs = []
    for k, horizon in enumerate(bundle.horizons):
        y_true = split.y_h_test[:, k]
        if len(np.unique(y_true)) < 2:
            metrics[f'horizon_{horizon}_auc'] = float('nan')
            continue
        auc = float(roc_auc_score(y_true, horizon_pred[:, k]))
        metrics[f'horizon_{horizon}_auc'] = auc
        aucs.append(auc)
    metrics['mean_horizon_auc'] = float(np.nanmean(aucs)) if aucs else float('nan')
    return metrics


def train_from_robomimic(path: str | Path, output_path: str | Path | None = None, config: AppConfig | None = None) -> tuple[VerifierBundle, dict, list[Episode]]:
    config = config or AppConfig()
    episodes = load_robomimic_hdf5(path, config=config.dataset)
    bundle, metrics = train_bundle_from_episodes(episodes, output_path=output_path, config=config)
    return bundle, metrics, episodes


def train_on_fixture(output_path: str | Path | None = None, fixture_path: str | Path | None = None, config: AppConfig | None = None) -> tuple[VerifierBundle, dict, Path]:
    config = config or AppConfig()
    fixture_path = Path(fixture_path) if fixture_path is not None else None
    if fixture_path is None:
        from .config import DEFAULT_FIXTURE_PATH
        fixture_path = DEFAULT_FIXTURE_PATH
    create_tiny_robomimic_fixture(fixture_path)
    bundle, metrics, _ = train_from_robomimic(fixture_path, output_path=output_path, config=config)
    return bundle, metrics, fixture_path
