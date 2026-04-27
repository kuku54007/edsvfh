from __future__ import annotations

import numpy as np

from .types import Episode, GroundTruthLabel


APPROACH, GRASP, LIFT, TRANSPORT, RELEASE = range(5)


def _parse_state(robot_state: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    rs = np.asarray(robot_state, dtype=np.float32).reshape(-1)
    padded = np.zeros(max(10, len(rs)), dtype=np.float32)
    padded[: len(rs)] = rs
    eef = padded[0:3]
    gripper = float(padded[3])
    obj = padded[4:7]
    goal = padded[7:10]
    joint = padded[10:]
    return eef, gripper, obj, goal, joint


def _has_object_goal_signal(obj: np.ndarray, goal: np.ndarray) -> bool:
    return bool(np.linalg.norm(obj) > 1e-6 or np.linalg.norm(goal) > 1e-6)


def _infer_object_goal_label(episode: Episode, idx: int) -> GroundTruthLabel:
    obs = episode.steps[idx].observation
    eef, gripper, obj, goal, _ = _parse_state(obs.robot_state)
    dist_eo = float(np.linalg.norm(eef - obj))
    dist_og = float(np.linalg.norm(obj - goal))
    obj_height = float(obj[2])
    gripper_closed = gripper < 0.45
    lifted = obj_height > 0.22
    near_goal = dist_og < 0.09

    if near_goal and gripper > 0.60:
        subgoal = RELEASE
        completion = float(np.clip(0.6 + 0.4 * gripper, 0.0, 1.0))
    elif near_goal and gripper_closed:
        subgoal = TRANSPORT
        completion = float(np.clip(1.0 - dist_og / 0.30, 0.0, 1.0))
    elif lifted and gripper_closed:
        if dist_og < 0.30:
            subgoal = TRANSPORT
            completion = float(np.clip(1.0 - dist_og / 0.50, 0.0, 1.0))
        else:
            subgoal = LIFT
            completion = float(np.clip((obj_height - 0.12) / 0.18, 0.0, 1.0))
    elif dist_eo < 0.08 and gripper_closed:
        subgoal = GRASP
        completion = float(np.clip(1.0 - dist_eo / 0.08, 0.0, 1.0))
    else:
        subgoal = APPROACH
        completion = float(np.clip(1.0 - dist_eo / 0.35, 0.0, 1.0))

    if idx + 1 < len(episode.steps):
        next_label = _infer_subgoal_only(episode.steps[idx + 1].observation.robot_state)
        done = int(next_label != subgoal)
    else:
        done = 1

    failure_onset = episode.failure_onset if episode.failure_onset is not None else len(episode.steps) + 100
    failure = int(episode.outcome == 'failure' and idx >= failure_onset)
    event = int(idx == 0 or done or obs.action_type in {'close_gripper', 'open_gripper', 'lift'} or (episode.failure_onset is not None and abs(idx - episode.failure_onset) <= 1))
    return GroundTruthLabel(
        subgoal=subgoal,
        completion=completion,
        done=done,
        failure=failure,
        failure_onset=failure_onset,
        action_type=obs.action_type,
        event=event,
    )


def _infer_proprio_only_label(episode: Episode, idx: int) -> GroundTruthLabel:
    obs = episode.steps[idx].observation
    eef, gripper, _, _, joint = _parse_state(obs.robot_state)
    prev_eef = eef if idx == 0 else _parse_state(episode.steps[idx - 1].observation.robot_state)[0]
    first_eef = _parse_state(episode.steps[0].observation.robot_state)[0]
    eef_delta = float(np.linalg.norm(eef - prev_eef))
    horizontal_disp = float(np.linalg.norm(eef[:2] - first_eef[:2]))
    height = float(eef[2])
    closed = gripper < 0.45
    opened = gripper > 0.60
    late_fraction = (idx + 1) / max(len(episode.steps), 1)

    if idx == len(episode.steps) - 1 and opened:
        subgoal = RELEASE
        completion = 1.0
    elif closed and height > first_eef[2] + 0.06:
        if late_fraction > 0.60 or horizontal_disp > 0.08:
            subgoal = TRANSPORT
            completion = float(np.clip(max(late_fraction, horizontal_disp / 0.20), 0.0, 1.0))
        else:
            subgoal = LIFT
            completion = float(np.clip((height - first_eef[2]) / 0.12, 0.0, 1.0))
    elif closed:
        subgoal = GRASP
        completion = float(np.clip(0.35 + late_fraction * 0.55 + min(eef_delta, 0.08), 0.0, 1.0))
    else:
        subgoal = APPROACH
        completion = float(np.clip(late_fraction * 0.65, 0.0, 0.85))

    next_subgoal = subgoal
    if idx + 1 < len(episode.steps):
        next_subgoal = _infer_subgoal_only(episode.steps[idx + 1].observation.robot_state)
    done = int(next_subgoal != subgoal or idx == len(episode.steps) - 1)

    failure_onset = episode.failure_onset if episode.failure_onset is not None else len(episode.steps) + 100
    failure = int(episode.outcome == 'failure' and idx >= failure_onset)
    uncertainty = float(obs.policy_stats[3]) if len(obs.policy_stats) >= 4 else 0.0
    event = int(
        idx == 0
        or done
        or obs.action_type in {'close_gripper', 'open_gripper', 'lift'}
        or uncertainty > 0.55
        or (episode.failure_onset is not None and abs(idx - episode.failure_onset) <= 1)
    )
    return GroundTruthLabel(
        subgoal=subgoal,
        completion=completion,
        done=done,
        failure=failure,
        failure_onset=failure_onset,
        action_type=obs.action_type,
        event=event,
    )


def infer_step_label(episode: Episode, idx: int) -> GroundTruthLabel:
    obs = episode.steps[idx].observation
    _, _, obj, goal, _ = _parse_state(obs.robot_state)
    if _has_object_goal_signal(obj, goal):
        return _infer_object_goal_label(episode, idx)
    return _infer_proprio_only_label(episode, idx)


def _infer_subgoal_only(robot_state: np.ndarray) -> int:
    eef, gripper, obj, goal, _ = _parse_state(robot_state)
    if _has_object_goal_signal(obj, goal):
        dist_eo = float(np.linalg.norm(eef - obj))
        dist_og = float(np.linalg.norm(obj - goal))
        obj_height = float(obj[2])
        if dist_og < 0.09 and gripper > 0.60:
            return RELEASE
        if obj_height > 0.22 and gripper < 0.45 and dist_og < 0.30:
            return TRANSPORT
        if obj_height > 0.22 and gripper < 0.45:
            return LIFT
        if dist_eo < 0.08 and gripper < 0.45:
            return GRASP
        return APPROACH

    if gripper > 0.60:
        return RELEASE
    if gripper < 0.45 and eef[2] > 0.18:
        return LIFT
    if gripper < 0.45:
        return GRASP
    return APPROACH


def infer_horizon_labels(episode: Episode, idx: int, horizons: tuple[int, ...]) -> np.ndarray:
    gt = infer_step_label(episode, idx)
    labels: list[int] = []
    for h in horizons:
        label = 0
        if episode.failure_onset is not None and episode.failure_onset > idx and (episode.failure_onset - idx) <= h:
            completion_idx = next((j for j in range(idx + 1, len(episode.steps)) if infer_step_label(episode, j).subgoal != gt.subgoal), len(episode.steps))
            if episode.failure_onset < completion_idx:
                label = 1
        labels.append(label)
    return np.asarray(labels, dtype=np.float32)
