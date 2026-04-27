from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field

from .types import StepObservation, VerificationOutput


class ObservationPayload(BaseModel):
    timestamp: int
    action_type: str
    robot_state: list[float]
    action: list[float]
    policy_stats: list[float] = Field(default_factory=list)
    image_png_b64: str | None = None
    instruction: str = ''

    def to_observation(self) -> StepObservation:
        image = None
        if self.image_png_b64:
            raw = base64.b64decode(self.image_png_b64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return StepObservation(
            image=image,
            robot_state=np.asarray(self.robot_state, dtype=np.float32),
            action=np.asarray(self.action, dtype=np.float32),
            policy_stats=np.asarray(self.policy_stats, dtype=np.float32) if self.policy_stats else np.zeros((3,), dtype=np.float32),
            action_type=self.action_type,
            timestamp=self.timestamp,
            instruction=self.instruction,
        )


class VerificationResponse(BaseModel):
    timestamp: int
    triggered: bool
    event_score: float
    visual_drift: float
    stall_score: float
    uncertainty_score: float
    high_stakes_score: float
    heartbeat_due: bool
    subgoal: str | None = None
    completion: float | None = None
    done_probability: float | None = None
    risk: list[float] | None = None
    decision: str | None = None
    reason: str | None = None
    terminated: bool = False
    terminal_decision: str | None = None
    termination_reason: str | None = None
    termination_timestamp: int | None = None
    post_termination: bool = False

    @classmethod
    def from_output(cls, out: VerificationOutput) -> 'VerificationResponse':
        return cls(
            timestamp=out.timestamp,
            triggered=out.triggered,
            event_score=out.event_score,
            visual_drift=out.visual_drift,
            stall_score=out.stall_score,
            uncertainty_score=out.uncertainty_score,
            high_stakes_score=out.high_stakes_score,
            heartbeat_due=out.heartbeat_due,
            subgoal=out.subgoal,
            completion=out.completion,
            done_probability=out.done_probability,
            risk=out.risk.tolist() if out.risk is not None else None,
            decision=out.decision,
            reason=out.reason,
            terminated=out.terminated,
            terminal_decision=out.terminal_decision,
            termination_reason=out.termination_reason,
            termination_timestamp=out.termination_timestamp,
            post_termination=out.post_termination,
        )


class MetricsResponse(BaseModel):
    metrics: dict[str, float]


class PipelineStatusResponse(BaseModel):
    terminated: bool
    termination_timestamp: int | None = None
    terminal_decision: str | None = None
    termination_reason: str | None = None
