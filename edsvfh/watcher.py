from __future__ import annotations

from collections import deque

import numpy as np

from .config import WatcherConfig
from .types import FeatureSnapshot


class EventWatcher:
    def __init__(self, config: WatcherConfig) -> None:
        self.config = config
        self.prev_visual: np.ndarray | None = None
        self.last_event_t: int | None = None
        self.progress_hist: deque[float] = deque(maxlen=config.stall_window)

    def reset(self) -> None:
        self.prev_visual = None
        self.last_event_t = None
        self.progress_hist.clear()

    def step(
        self,
        timestamp: int,
        snapshot: FeatureSnapshot,
        progress_proxy: float,
        policy_uncertainty: float,
        action_type: str,
    ) -> dict[str, float | bool]:
        if self.prev_visual is None:
            visual_drift = 1.0
        else:
            a = self.prev_visual
            b = snapshot.visual_embedding
            denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
            visual_drift = float(np.clip(1.0 - np.dot(a, b) / denom, 0.0, 1.0))

        self.progress_hist.append(progress_proxy)
        if len(self.progress_hist) >= 3:
            stall = 1.0 if (max(self.progress_hist) - min(self.progress_hist)) < self.config.stall_delta_threshold else 0.0
        else:
            stall = 0.0

        uncertainty = float(np.clip(policy_uncertainty, 0.0, 1.0))
        high_stakes = 1.0 if action_type in {"close_gripper", "open_gripper", "lift"} else 0.0
        score = (
            self.config.visual_weight * visual_drift
            + self.config.stall_weight * stall
            + self.config.uncertainty_weight * uncertainty
            + self.config.high_stakes_weight * high_stakes
        )
        heartbeat_due = self.last_event_t is None or (timestamp - self.last_event_t) >= self.config.heartbeat_steps
        trigger = bool(score > self.config.trigger_threshold or heartbeat_due)
        if trigger:
            self.last_event_t = timestamp
        self.prev_visual = snapshot.visual_embedding.copy()
        return {
            "trigger": trigger,
            "score": score,
            "visual_drift": visual_drift,
            "stall": stall,
            "uncertainty": uncertainty,
            "high_stakes": high_stakes,
            "heartbeat_due": heartbeat_due,
        }
