from __future__ import annotations

import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from edsvfh.droid_convert import MockDroidEpisodeSource, _standardize_droid_episode
from edsvfh.droid_failure import discover_droid_raw_failure_episodes, generate_droid_failure_manifest_from_episode_source, generate_droid_failure_manifest_from_raw, infer_droid_raw_outcome, infer_droid_raw_outcome_from_path


def _write_mock_mp4(path: Path, *, n: int = 8, size: int = 48) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (size, size))
    assert writer.isOpened()
    for i in range(n):
        img = np.full((size, size, 3), 220, dtype=np.uint8)
        cv2.circle(img, (8 + 4 * i, 24), 5, (0, 0, 220), -1)
        writer.write(img)
    writer.release()


def _write_mock_raw_failure_episode(root: Path, episode_name: str = "episode_000") -> Path:
    ep_dir = root / "1.0.1" / "AUTOLab" / "failure" / "2023-07-31" / episode_name
    ep_dir.mkdir(parents=True, exist_ok=True)
    n = 8
    with h5py.File(ep_dir / "trajectory.h5", "w") as h5:
        obs = h5.create_group("observation")
        obs.create_dataset("cartesian_position", data=np.linspace(0, 1, n * 6, dtype=np.float32).reshape(n, 6))
        obs.create_dataset("gripper_position", data=np.linspace(0, 1, n, dtype=np.float32).reshape(n, 1))
        obs.create_dataset("joint_position", data=np.linspace(0, 1, n * 7, dtype=np.float32).reshape(n, 7))
        action = h5.create_group("action")
        action.create_dataset("cartesian_position", data=np.linspace(0, 1, n * 7, dtype=np.float32).reshape(n, 7))
    (ep_dir / "metadata_AUTOLab+mock.json").write_text(
        json.dumps({"task": "pick_place", "language_instruction": "pick and place the object"}),
        encoding="utf-8",
    )
    _write_mock_mp4(ep_dir / "recordings" / "MP4" / "exterior_image_1_left.mp4", n=n)
    return ep_dir




def _write_mock_raw_not_success_metadata_episode(root: Path, episode_name: str = "episode_001") -> Path:
    ep_dir = root / "1.0.1" / "AUTOLab" / "2023-07-31" / episode_name
    ep_dir.mkdir(parents=True, exist_ok=True)
    n = 6
    with h5py.File(ep_dir / "trajectory.h5", "w") as h5:
        obs = h5.create_group("observation")
        obs.create_dataset("cartesian_position", data=np.linspace(0, 1, n * 6, dtype=np.float32).reshape(n, 6))
        obs.create_dataset("gripper_position", data=np.linspace(0, 1, n, dtype=np.float32).reshape(n, 1))
        action = h5.create_group("action")
        action.create_dataset("cartesian_position", data=np.linspace(0, 1, n * 7, dtype=np.float32).reshape(n, 7))
    (ep_dir / "metadata_AUTOLab+mock.json").write_text(
        json.dumps({"task": "pick_place", "language_instruction": "pick and place the object", "success": False}),
        encoding="utf-8",
    )
    _write_mock_mp4(ep_dir / "recordings" / "MP4" / "exterior_image_1_left.mp4", n=n)
    return ep_dir


def test_infer_droid_raw_outcome() -> None:
    assert infer_droid_raw_outcome({"success": False}) == "failure"
    assert infer_droid_raw_outcome({"is_success": True}) == "success"
    assert infer_droid_raw_outcome({"outcome": "not_successful"}) == "failure"
    assert infer_droid_raw_outcome({}, "/tmp/droid/failure/demo") == "failure"
    assert infer_droid_raw_outcome({}, "/tmp/droid/unknown/demo") == "unknown"


def test_infer_droid_raw_outcome_from_path() -> None:
    assert infer_droid_raw_outcome_from_path("/tmp/droid/1.0.1/AUTOLab/failure/demo") == "failure"
    assert infer_droid_raw_outcome_from_path("/tmp/droid/1.0.1/AUTOLab/not_successful/demo") == "failure"
    assert infer_droid_raw_outcome_from_path("/tmp/droid/1.0.1/AUTOLab/success/demo") == "success"
    assert infer_droid_raw_outcome_from_path("/tmp/droid/1.0.1/AUTOLab/unknown/demo") == "unknown"


def test_discover_droid_raw_failure_episodes_uses_metadata(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    ep_dir = _write_mock_raw_not_success_metadata_episode(raw_root)
    episodes = discover_droid_raw_failure_episodes(raw_root)
    assert episodes == [ep_dir]


def test_generate_droid_failure_manifest_from_raw(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _write_mock_raw_failure_episode(raw_root)
    manifest_path = tmp_path / "manifest.jsonl"
    frames_root = tmp_path / "frames"

    result = generate_droid_failure_manifest_from_raw(
        raw_root,
        manifest_path,
        frames_root=frames_root,
        image_size=32,
        frame_stride=2,
        max_episodes=1,
        show_progress=False,
    )

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert result.failure_episodes == 1
    assert len(rows) == 1
    row = rows[0]
    assert row["outcome"] == "failure"
    assert row["failure_onset"] is None
    assert row["source_format"] == "droid_raw_not_successful"
    assert row["failure_label_source"] == "droid_raw_failure_folder"
    assert Path(row["frames_dir"]).exists()
    assert len(list(Path(row["frames_dir"]).glob("*.png"))) == 4
    assert np.load(row["eef_npy"]).shape == (4, 3)
    assert np.load(row["gripper_npy"]).shape == (4, 1)
    assert np.load(row["action_npy"]).shape[0] == 4
    assert np.load(row["policy_uncertainty_npy"]).shape == (4, 1)


def test_droid_rlds_converter_infers_failure_from_metadata_path() -> None:
    steps = []
    for t in range(4):
        steps.append({
            "language_instruction": "move the object",
            "reward": np.float32(0.0),
            "action": np.zeros(7, dtype=np.float32),
            "observation": {
                "cartesian_position": np.zeros(6, dtype=np.float32),
                "gripper_position": np.zeros(1, dtype=np.float32),
                "joint_position": np.zeros(7, dtype=np.float32),
                "exterior_image_1_left": np.zeros((16, 16, 3), dtype=np.uint8),
            },
            "is_last": np.bool_(t == 3),
        })
    episode = _standardize_droid_episode(
        {
            "episode_metadata": {"file_path": "/tmp/droid/failure/episode_000", "success": False},
            "steps": steps,
            "episode_index": 0,
        },
        image_size=16,
        step_stride=1,
        action_space="raw_action",
        camera_preference=("exterior_image_1_left",),
    )
    assert episode["outcome"] == "failure"
    assert episode["failure_onset"] is None



def test_generate_droid_failure_manifest_from_rlds_episode_source(tmp_path: Path) -> None:
    source = MockDroidEpisodeSource(num_episodes=10, steps_per_episode=8, image_size=24, include_failures=True, seed=123)
    manifest_path = tmp_path / "rlds_failure_manifest.jsonl"
    frames_root = tmp_path / "rlds_frames"

    result = generate_droid_failure_manifest_from_episode_source(
        source,
        manifest_path,
        source_root="mock://droid_101",
        frames_root=frames_root,
        image_size=24,
        frame_stride=2,
        max_episodes=2,
        show_progress=False,
    )

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert result.failure_episodes == 2
    assert len(rows) == 2
    assert all(row["outcome"] == "failure" for row in rows)
    assert all(row["source_format"] == "droid_rlds_not_successful" for row in rows)
    assert all(row["failure_label_source"] == "droid_rlds_metadata_path" for row in rows)
    for row in rows:
        frame_dir = Path(row["frames_dir"])
        assert frame_dir.exists()
        assert len(list(frame_dir.glob("*.png"))) == 4
        assert np.load(row["eef_npy"]).shape == (4, 3)
        assert np.load(row["gripper_npy"]).shape == (4, 1)
        assert np.load(row["action_npy"]).shape[0] == 4
        assert np.load(row["policy_uncertainty_npy"]).shape == (4, 1)
