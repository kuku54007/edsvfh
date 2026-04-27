from __future__ import annotations

from collections import deque

import numpy as np

from .memory import EventMemory
from .types import FeatureSnapshot


class ContextBuilder:
    def __init__(self, window: int) -> None:
        self.window = window
        self.recent: deque[FeatureSnapshot] = deque(maxlen=window)

    def reset(self) -> None:
        self.recent.clear()

    def update(self, snapshot: FeatureSnapshot) -> None:
        self.recent.append(snapshot)

    def heuristic_progress(self) -> float:
        if not self.recent:
            return 0.0
        snap = self.recent[-1]
        progress = (
            0.45 * (1.0 - min(1.0, snap.object_gripper_dist / 0.6))
            + 0.25 * np.clip(snap.object_height, 0.0, 1.0)
            + 0.30 * (1.0 - min(1.0, snap.object_target_dist / 0.8))
        )
        return float(np.clip(progress, 0.0, 1.0))

    def build(self, timestamp: int, current: FeatureSnapshot, memory: EventMemory) -> np.ndarray:
        recent_mat = np.stack([snap.vector for snap in self.recent], axis=0) if self.recent else current.vector[None, :]
        mean = recent_mat.mean(axis=0)
        std = recent_mat.std(axis=0)
        delta = recent_mat[-1] - recent_mat[0]
        memory_summary = memory.summary(timestamp)
        return np.concatenate([current.vector, mean, std, delta, memory_summary], dtype=np.float32)
