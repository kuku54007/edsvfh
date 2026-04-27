from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np

from .config import DatasetConfig
from .types import ACTION_TYPES, Episode, EpisodeStep, StepObservation


@dataclass(frozen=True)
class PublicDatasetSpec:
    name: str
    kind: str
    recommended_role: str
    download_page: str
    note: str


DATASET_CATALOG: tuple[PublicDatasetSpec, ...] = (
    PublicDatasetSpec(
        name='DROID',
        kind='real robot, RLDS/raw',
        recommended_role='primary large-scale success training',
        download_page='https://droid-dataset.github.io/',
        note='Large real-world manipulation corpus with RLDS and raw releases.',
    ),
    PublicDatasetSpec(
        name='BridgeData V2',
        kind='real robot, TFDS/RLDS or raw',
        recommended_role='cross-environment generalization training',
        download_page='https://bridgedata-v2.github.io/',
        note='Natural-language, multi-environment manipulation trajectories.',
    ),
    PublicDatasetSpec(
        name='LIBERO',
        kind='benchmark suites',
        recommended_role='structured benchmark validation',
        download_page='https://github.com/Lifelong-Robot-Learning/LIBERO',
        note='Task suites and human demonstrations for long-horizon evaluation.',
    ),
    PublicDatasetSpec(
        name='CALVIN',
        kind='language-conditioned benchmark',
        recommended_role='long-horizon benchmark validation',
        download_page='https://github.com/mees/calvin',
        note='Long-horizon language-conditioned manipulation benchmark.',
    ),
    PublicDatasetSpec(
        name='robomimic',
        kind='HDF5 benchmark datasets',
        recommended_role='easy-to-adapt public format / ablation',
        download_page='https://robomimic.github.io/docs/datasets/overview.html',
        note='Well-documented HDF5 schema with simulation and some real data.',
    ),
    PublicDatasetSpec(
        name='FAILURE (FINO)',
        kind='failure detection dataset',
        recommended_role='failure supervision / horizon validation',
        download_page='https://sites.google.com/view/fino-net/failure-dataset',
        note='Public failure dataset for place, pour, put-in, put-on, and push tasks.',
    ),
    PublicDatasetSpec(
        name='FailGen (AHA / RLBench)',
        kind='procedurally generated failures',
        recommended_role='large-scale failure augmentation',
        download_page='https://github.com/annahung31/AHA',
        note='Synthetic failure generation across RLBench tasks.',
    ),
    PublicDatasetSpec(
        name='REASSEMBLE',
        kind='contact-rich multimodal assembly dataset',
        recommended_role='multimodal validation / contact-rich tasks',
        download_page='https://reassemble-dataset.github.io/',
        note='Assembly/disassembly dataset with RGB, force-torque, microphones, and more.',
    ),
)


def dataset_catalog() -> list[dict[str, str]]:
    return [spec.__dict__.copy() for spec in DATASET_CATALOG]


def infer_action_type(action: np.ndarray, current_gripper: float, previous_gripper: float | None = None) -> str:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    move_mag = float(np.linalg.norm(action[:3])) if len(action) >= 3 else float(np.linalg.norm(action))
    grip_delta = 0.0 if previous_gripper is None else float(current_gripper - previous_gripper)
    if grip_delta < -0.15:
        return 'close_gripper'
    if grip_delta > 0.15:
        return 'open_gripper'
    if len(action) >= 3 and action[2] > 0.05:
        return 'lift'
    if move_mag > 0.08:
        return 'transport'
    if move_mag > 0.01:
        return 'move'
    return 'idle'


def _find_first_key(keys: list[str], candidates: list[str]) -> str | None:
    lowered = {k.lower(): k for k in keys}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    for key in keys:
        low = key.lower()
        if any(cand.lower() in low for cand in candidates):
            return key
    return None


def _read_array(group: h5py.Group, key: str | None, default_shape: tuple[int, ...], idx: int) -> np.ndarray:
    if key is None or key not in group:
        return np.zeros(default_shape, dtype=np.float32)
    value = np.asarray(group[key][idx])
    if value.dtype.kind in {'u', 'i'}:
        return value.astype(np.float32)
    return value.astype(np.float32)


def load_robomimic_hdf5(path: str | Path, config: DatasetConfig | None = None) -> list[Episode]:
    config = config or DatasetConfig()
    path = Path(path)
    episodes: list[Episode] = []
    with h5py.File(path, 'r') as f:
        data_group = f['data']
        demo_keys = sorted(data_group.keys())
        if config.max_episodes is not None:
            demo_keys = demo_keys[: config.max_episodes]
        for demo_key in demo_keys:
            demo = data_group[demo_key]
            obs_group = demo['obs']
            next_obs_group = demo['next_obs'] if 'next_obs' in demo else obs_group
            obs_keys = list(obs_group.keys())
            image_key = config.image_key or _find_first_key(obs_keys, ['agentview_image', 'image', 'rgb'])
            eef_key = config.eef_key or _find_first_key(obs_keys, ['robot0_eef_pos', 'eef_pos', 'eef'])
            gripper_key = config.gripper_key or _find_first_key(obs_keys, ['robot0_gripper_qpos', 'gripper_qpos', 'gripper'])
            object_key = config.object_key or _find_first_key(obs_keys, ['object_pos', 'object-state', 'object'])
            goal_key = config.goal_key or _find_first_key(obs_keys, ['goal_pos', 'goal'])
            joint_key = _find_first_key(obs_keys, ['robot0_joint_pos', 'joint_position', 'joint_pos'])
            uncertainty_key = _find_first_key(obs_keys, ['policy_uncertainty', 'uncertainty'])
            pre_vec_key = _find_first_key(obs_keys, ['precomputed_vector'])
            pre_vis_key = _find_first_key(obs_keys, ['precomputed_visual_embedding'])
            pre_oh_key = _find_first_key(obs_keys, ['precomputed_action_one_hot'])
            pre_ogd_key = _find_first_key(obs_keys, ['precomputed_object_gripper_dist'])
            pre_otd_key = _find_first_key(obs_keys, ['precomputed_object_target_dist'])
            pre_ohgt_key = _find_first_key(obs_keys, ['precomputed_object_height'])
            pre_visib_key = _find_first_key(obs_keys, ['precomputed_visibility'])
            actions = np.asarray(demo['actions'])
            dones = np.asarray(demo['dones']) if 'dones' in demo else np.zeros((len(actions),), dtype=np.int32)
            rewards = np.asarray(demo['rewards']) if 'rewards' in demo else np.zeros((len(actions),), dtype=np.float32)
            task = str(demo.attrs.get('task', 'robomimic_task'))
            instruction = str(demo.attrs.get('instruction', config.default_instruction))
            outcome = str(demo.attrs.get('outcome', 'success'))
            failure_onset = demo.attrs.get('failure_onset', None)
            if failure_onset is not None:
                failure_onset = int(failure_onset)
            original_failure_onset = demo.attrs.get('original_failure_onset', None)
            if original_failure_onset is not None:
                original_failure_onset = int(original_failure_onset)
            pseudo_failure_onset = demo.attrs.get('pseudo_failure_onset', None)
            if pseudo_failure_onset is not None:
                pseudo_failure_onset = int(pseudo_failure_onset)
            steps: list[EpisodeStep] = []
            prev_gripper: float | None = None
            for i in range(len(actions)):
                image = np.asarray(obs_group[image_key][i]).astype(np.uint8) if image_key is not None else None
                eef = _read_array(obs_group, eef_key, (3,), i).reshape(-1)
                eef = np.pad(eef[:3], (0, max(0, 3 - len(eef))))[:3]
                gr = _read_array(obs_group, gripper_key, (1,), i).reshape(-1)
                gripper = float(gr[0]) if len(gr) > 0 else 0.0
                obj = _read_array(obs_group, object_key, (3,), i).reshape(-1)
                if len(obj) == 0:
                    obj = np.zeros(3, dtype=np.float32)
                if len(obj) < 3:
                    obj = np.pad(obj, (0, 3 - len(obj)))
                obj = obj[:3]
                goal = _read_array(obs_group, goal_key, (3,), i).reshape(-1)
                if len(goal) == 0 and goal_key is None and 'goal_pos' in demo.attrs:
                    goal = np.asarray(demo.attrs['goal_pos'], dtype=np.float32).reshape(-1)
                if len(goal) == 0:
                    goal = np.zeros(3, dtype=np.float32)
                if len(goal) < 3:
                    goal = np.pad(goal, (0, 3 - len(goal)))
                goal = goal[:3]
                joint = _read_array(obs_group, joint_key, (0,), i).reshape(-1) if joint_key is not None else np.zeros((0,), dtype=np.float32)
                robot_state = np.concatenate([eef, np.array([gripper], dtype=np.float32), obj.astype(np.float32), goal.astype(np.float32), joint.astype(np.float32)], dtype=np.float32)
                action = actions[i].astype(np.float32).reshape(-1)
                action_type = infer_action_type(action, gripper, prev_gripper)
                prev_gripper = gripper
                policy_uncertainty = float(_read_array(obs_group, uncertainty_key, (1,), i).reshape(-1)[0]) if uncertainty_key is not None else 0.0
                policy_stats = np.array(
                    [float(min(1.0, np.linalg.norm(action))), float(abs(action[-1]) if len(action) else 0.0), 1.0, policy_uncertainty],
                    dtype=np.float32,
                )
                precomputed_vector = np.asarray(obs_group[pre_vec_key][i], dtype=np.float32) if pre_vec_key is not None else None
                precomputed_visual_embedding = np.asarray(obs_group[pre_vis_key][i], dtype=np.float32) if pre_vis_key is not None else None
                precomputed_action_one_hot = np.asarray(obs_group[pre_oh_key][i], dtype=np.float32) if pre_oh_key is not None else None
                precomputed_object_gripper_dist = float(np.asarray(obs_group[pre_ogd_key][i]).reshape(-1)[0]) if pre_ogd_key is not None else None
                precomputed_object_target_dist = float(np.asarray(obs_group[pre_otd_key][i]).reshape(-1)[0]) if pre_otd_key is not None else None
                precomputed_object_height = float(np.asarray(obs_group[pre_ohgt_key][i]).reshape(-1)[0]) if pre_ohgt_key is not None else None
                precomputed_visibility = float(np.asarray(obs_group[pre_visib_key][i]).reshape(-1)[0]) if pre_visib_key is not None else None
                obs = StepObservation(
                    image=image,
                    robot_state=robot_state,
                    action=action,
                    policy_stats=policy_stats,
                    action_type=action_type,
                    timestamp=i,
                    instruction=instruction,
                    precomputed_vector=precomputed_vector,
                    precomputed_visual_embedding=precomputed_visual_embedding,
                    precomputed_action_one_hot=precomputed_action_one_hot,
                    precomputed_object_gripper_dist=precomputed_object_gripper_dist,
                    precomputed_object_target_dist=precomputed_object_target_dist,
                    precomputed_object_height=precomputed_object_height,
                    precomputed_visibility=precomputed_visibility,
                )
                aux = {
                    'reward': float(rewards[i]),
                    'done': int(dones[i]),
                    'demo_key': demo_key,
                }
                if failure_onset is not None:
                    aux['failure_onset'] = failure_onset
                if original_failure_onset is not None:
                    aux['original_failure_onset'] = original_failure_onset
                if pseudo_failure_onset is not None:
                    aux['pseudo_failure_onset'] = pseudo_failure_onset
                steps.append(EpisodeStep(observation=obs, aux=aux))
            episodes.append(Episode(task=task, instruction=instruction, steps=steps, outcome=outcome, failure_onset=failure_onset, source=str(path)))
    return episodes


def load_lerobot_info_json(path: str | Path) -> dict:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _render_step_image(eef: np.ndarray, obj: np.ndarray, goal: np.ndarray, gripper: float, size: int = 96) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (240, 240, 240)
    def to_px(p: np.ndarray) -> tuple[int, int]:
        x = int(np.clip(p[0], 0.0, 1.0) * (size - 1))
        y = int((1.0 - np.clip(p[1], 0.0, 1.0)) * (size - 1))
        return x, y
    cv2.circle(img, to_px(goal), 8, (0, 180, 0), -1)
    cv2.circle(img, to_px(obj), 7, (0, 0, 220), -1)
    grip_color = (220, 0, 0) if gripper < 0.5 else (255, 120, 0)
    cv2.circle(img, to_px(eef), 6, grip_color, -1)
    cv2.line(img, to_px(eef), to_px(obj), (100, 100, 100), 1)
    return img


def _generate_pick_place_episode(kind: str, seed: int, length: int = 24) -> tuple[list[EpisodeStep], int | None]:
    rng = np.random.default_rng(seed)
    obj = np.array([0.30, 0.35, 0.05], dtype=np.float32)
    goal = np.array([0.78, 0.72, 0.05], dtype=np.float32)
    eef = np.array([0.10, 0.20, 0.18], dtype=np.float32)
    gripper = 1.0
    steps: list[EpisodeStep] = []
    failure_onset: int | None = None
    carried = False
    for t in range(length):
        action = np.zeros(4, dtype=np.float32)
        policy_uncertainty = 0.05
        if t < 6:
            target = np.array([obj[0], obj[1], 0.14], dtype=np.float32)
            eef += 0.35 * (target - eef)
            action[:3] = target - eef
            action_type = 'move'
        elif t == 6:
            gripper = 0.0
            if kind != 'miss_grasp':
                carried = True
            else:
                failure_onset = 6
                policy_uncertainty = 0.98
            action_type = 'close_gripper'
            action[3] = -1.0
        elif 7 <= t < 11:
            target = np.array([eef[0], eef[1], 0.40], dtype=np.float32)
            eef += 0.45 * (target - eef)
            if carried:
                obj = np.array([eef[0], eef[1], max(obj[2], eef[2] - 0.10)], dtype=np.float32)
            action[:3] = target - eef
            action_type = 'lift'
        elif 11 <= t < 18:
            target = np.array([goal[0], goal[1], 0.40], dtype=np.float32)
            eef += 0.28 * (target - eef)
            if carried:
                obj = np.array([eef[0], eef[1], max(obj[2], eef[2] - 0.10)], dtype=np.float32)
            if kind == 'drift' and t >= 12:
                policy_uncertainty = 0.55 if t == 12 else 0.75
            if kind == 'drift' and t >= 13:
                obj += np.array([0.06, -0.05, 0.0], dtype=np.float32)
                if failure_onset is None:
                    failure_onset = 13
            action[:3] = target - eef
            action_type = 'transport'
        elif t == 18:
            if kind == 'slip':
                carried = False
                obj += np.array([-0.12, 0.02, -0.03], dtype=np.float32)
                failure_onset = 18
                policy_uncertainty = 0.92
            gripper = 1.0
            action[3] = 1.0
            action_type = 'open_gripper'
            if carried:
                obj = goal.copy()
        else:
            action_type = 'idle'
            action *= 0.0
        if carried and kind not in {'drift', 'slip'} and 11 <= t < 18:
            obj = np.array([eef[0], eef[1], max(0.28, eef[2] - 0.10)], dtype=np.float32)
        image = _render_step_image(eef[:2], obj[:2], goal[:2], gripper)
        robot_state = np.concatenate([eef, np.array([gripper], dtype=np.float32), obj, goal], dtype=np.float32)
        policy_stats = np.array([float(min(1.0, np.linalg.norm(action[:3]))), float(abs(action[3])), 1.0, policy_uncertainty], dtype=np.float32)
        obs = StepObservation(
            image=image,
            robot_state=robot_state,
            action=action,
            policy_stats=policy_stats,
            action_type=action_type,
            timestamp=t,
            instruction='Pick the object and place it at the goal.',
        )
        steps.append(EpisodeStep(observation=obs, aux={'kind': kind}))
    return steps, failure_onset


def create_tiny_robomimic_fixture(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    episodes = [
        ('success', 7), ('success', 11), ('success', 23), ('success', 29), ('success', 31), ('success', 37), ('success', 41), ('success', 43),
        ('miss_grasp', 13), ('miss_grasp', 47), ('miss_grasp', 53), ('miss_grasp', 59),
        ('drift', 17), ('drift', 61), ('drift', 67), ('drift', 71),
        ('slip', 19), ('slip', 73), ('slip', 79), ('slip', 83),
    ]
    with h5py.File(path, 'w') as f:
        data = f.create_group('data')
        for idx, (kind, seed) in enumerate(episodes):
            steps, failure_onset = _generate_pick_place_episode(kind, seed)
            grp = data.create_group(f'demo_{idx:03d}')
            T = len(steps)
            actions = np.stack([s.observation.action for s in steps], axis=0).astype(np.float32)
            dones = np.zeros((T,), dtype=np.int32)
            dones[-1] = 1
            rewards = np.zeros((T,), dtype=np.float32)
            if failure_onset is None:
                rewards[-1] = 1.0
            grp.create_dataset('actions', data=actions)
            grp.create_dataset('dones', data=dones)
            grp.create_dataset('rewards', data=rewards)
            obs_g = grp.create_group('obs')
            nxt_g = grp.create_group('next_obs')
            images = np.stack([s.observation.image for s in steps], axis=0).astype(np.uint8)
            eef = np.stack([s.observation.robot_state[0:3] for s in steps], axis=0).astype(np.float32)
            gripper = np.stack([np.array([s.observation.robot_state[3]], dtype=np.float32) for s in steps], axis=0)
            obj = np.stack([s.observation.robot_state[4:7] for s in steps], axis=0).astype(np.float32)
            goal = np.stack([s.observation.robot_state[7:10] for s in steps], axis=0).astype(np.float32)
            uncertainty = np.stack([np.array([s.observation.policy_stats[3]], dtype=np.float32) for s in steps], axis=0)
            for parent in (obs_g, nxt_g):
                parent.create_dataset('agentview_image', data=images)
                parent.create_dataset('robot0_eef_pos', data=eef)
                parent.create_dataset('robot0_gripper_qpos', data=gripper)
                parent.create_dataset('object_pos', data=obj)
                parent.create_dataset('goal_pos', data=goal)
                parent.create_dataset('policy_uncertainty', data=uncertainty)
            grp.attrs['task'] = 'pick_place'
            grp.attrs['instruction'] = 'Pick the object and place it at the green goal.'
            grp.attrs['outcome'] = 'success' if failure_onset is None else 'failure'
            if failure_onset is not None:
                grp.attrs['failure_onset'] = failure_onset
    return path


def list_hdf5_shards(path: str | Path, split: str | None = None) -> list[Path]:
    root = Path(path)
    if root.is_file() and root.suffix.lower() in {'.hdf5', '.h5'}:
        return [root]
    base = root / split if split is not None and (root / split).exists() else root
    return sorted([p for p in base.glob('*.hdf5') if p.is_file()])
