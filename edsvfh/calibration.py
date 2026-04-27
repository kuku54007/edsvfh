from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Binary Platt scaling on top of raw model probabilities."""

    def __init__(self, model: LogisticRegression | None = None, constant: float | None = None) -> None:
        self.model = model
        self.constant = constant

    @classmethod
    def fit(cls, base_prob: np.ndarray, labels: np.ndarray, seed: int = 0) -> "PlattCalibrator":
        labels = labels.astype(int)
        if len(np.unique(labels)) < 2:
            return cls(model=None, constant=float(labels[0]))
        prob = np.clip(base_prob.astype(float), 1e-6, 1.0 - 1e-6)
        logits = np.log(prob / (1.0 - prob)).reshape(-1, 1)
        model = LogisticRegression(random_state=seed, max_iter=200)
        model.fit(logits, labels)
        return cls(model=model)

    def predict(self, base_prob: np.ndarray) -> np.ndarray:
        prob = np.clip(np.asarray(base_prob, dtype=float), 1e-6, 1.0 - 1e-6)
        if self.model is None:
            assert self.constant is not None
            return np.full(prob.shape, self.constant, dtype=float)
        logits = np.log(prob / (1.0 - prob)).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]
