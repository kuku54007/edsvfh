from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SUBGOALS = ["approach", "grasp", "lift", "transport", "release"]
ACTION_TYPES = ["move", "close_gripper", "lift", "transport", "open_gripper", "idle"]


@dataclass(slots=True)
class StepObservation:
    image: np.ndarray | None
    robot_state: np.ndarray
    action: np.ndarray
    policy_stats: np.ndarray
    action_type: str
    timestamp: int
    instruction: str = ""
    precomputed_vector: np.ndarray | None = None
    precomputed_visual_embedding: np.ndarray | None = None
    precomputed_action_one_hot: np.ndarray | None = None
    precomputed_object_gripper_dist: float | None = None
    precomputed_object_target_dist: float | None = None
    precomputed_object_height: float | None = None
    precomputed_visibility: float | None = None


@dataclass(slots=True)
class EpisodeStep:
    observation: StepObservation
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Episode:
    task: str
    instruction: str
    steps: list[EpisodeStep]
    outcome: str = "unknown"  # success | failure | unknown
    failure_onset: int | None = None
    source: str = ""


@dataclass(slots=True)
class GroundTruthLabel:
    subgoal: int
    completion: float
    done: int
    failure: int
    failure_onset: int
    action_type: str
    event: int


@dataclass(slots=True)
class FeatureSnapshot:
    vector: np.ndarray
    visual_embedding: np.ndarray
    object_gripper_dist: float
    object_target_dist: float
    object_height: float
    visibility: float
    action_one_hot: np.ndarray


@dataclass(slots=True)
class EventPacket:
    timestamp: int
    subgoal: int
    completion: float
    done: float
    risk: np.ndarray
    visual_embedding: np.ndarray
    note: str = ""


@dataclass(slots=True)
class VerificationOutput:
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
    risk: np.ndarray | None = None
    decision: str | None = None
    reason: str | None = None
    terminated: bool = False
    terminal_decision: str | None = None
    termination_reason: str | None = None
    termination_timestamp: int | None = None
    post_termination: bool = False
