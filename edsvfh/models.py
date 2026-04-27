from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

from .calibration import PlattCalibrator
from .types import SUBGOALS


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def adapt_feature_dim(x: np.ndarray, target_dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[1] == target_dim:
        return x
    if x.shape[1] < target_dim:
        pad = np.zeros((x.shape[0], target_dim - x.shape[1]), dtype=np.float32)
        return np.concatenate([x, pad], axis=1)
    return x[:, :target_dim]


def _predict_binary_probability(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(x)
        prob = np.asarray(prob, dtype=np.float64)
        if prob.ndim == 1:
            return prob
        if prob.shape[1] == 1:
            return prob[:, 0]
        return prob[:, 1]
    if hasattr(model, 'decision_function'):
        return _sigmoid(np.asarray(model.decision_function(x), dtype=np.float64))
    pred = np.asarray(model.predict(x), dtype=np.float64)
    return np.clip(pred, 0.0, 1.0)


@dataclass
class VerifierBundle:
    subgoal_model: object
    completion_model: object
    done_model: object
    horizon_models: list[object | None]
    horizon_calibrators: list[PlattCalibrator]
    horizons: tuple[int, ...]
    input_dim: int
    num_subgoals: int
    metadata: dict
    feature_scaler: object | None = None

    def _transform(self, x: np.ndarray) -> np.ndarray:
        x = adapt_feature_dim(x, self.input_dim)
        if self.feature_scaler is not None:
            return self.feature_scaler.transform(x)
        return x

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = self._transform(x)
        subgoal_prob = np.asarray(self.subgoal_model.predict_proba(x), dtype=np.float64)
        subgoal_pred = subgoal_prob.argmax(axis=1)
        completion = np.clip(np.asarray(self.completion_model.predict(x), dtype=np.float64), 0.0, 1.0)
        done = _predict_binary_probability(self.done_model, x)
        horizon_prob: list[np.ndarray] = []
        for model, calibrator in zip(self.horizon_models, self.horizon_calibrators):
            if model is None:
                prob = calibrator.predict(np.full((len(x),), 0.5, dtype=float))
            else:
                base_prob = _predict_binary_probability(model, x)
                prob = calibrator.predict(base_prob)
            horizon_prob.append(prob)
        horizon = np.stack(horizon_prob, axis=1)
        # Practical monotonic projection matching the methodology's monotonicity intent.
        horizon = np.maximum.accumulate(horizon, axis=1)
        return subgoal_pred, completion, done, horizon

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> 'VerifierBundle':
        with Path(path).open('rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f'Unexpected bundle type: {type(obj)!r}')
        return obj

    def describe(self) -> dict:
        return {
            'num_subgoals': self.num_subgoals,
            'subgoals': list(SUBGOALS[: self.num_subgoals]),
            'horizons': list(self.horizons),
            'input_dim': self.input_dim,
            'feature_scaler': type(self.feature_scaler).__name__ if self.feature_scaler is not None else None,
            **self.metadata,
        }
