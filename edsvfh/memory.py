from __future__ import annotations

from collections import deque

import numpy as np

from .types import EventPacket


class EventMemory:
    def __init__(self, capacity: int, num_subgoals: int, num_horizons: int) -> None:
        self.capacity = capacity
        self.num_subgoals = num_subgoals
        self.num_horizons = num_horizons
        self.events: deque[EventPacket] = deque(maxlen=capacity)

    def reset(self) -> None:
        self.events.clear()

    def add(self, packet: EventPacket) -> None:
        self.events.append(packet)

    def summary(self, timestamp: int) -> np.ndarray:
        if not self.events:
            return np.zeros(self.num_subgoals + 2 + self.num_horizons + 1, dtype=np.float32)
        last = self.events[-1]
        subgoal_one_hot = np.zeros(self.num_subgoals, dtype=np.float32)
        subgoal_one_hot[last.subgoal] = 1.0
        delta_t = min(1.0, float(timestamp - last.timestamp) / 20.0)
        return np.concatenate(
            [
                subgoal_one_hot,
                np.array([last.completion, last.done], dtype=np.float32),
                last.risk.astype(np.float32),
                np.array([delta_t], dtype=np.float32),
            ],
            dtype=np.float32,
        )
