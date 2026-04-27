from __future__ import annotations

import numpy as np
from pathlib import Path

from edsvfh.config import AppConfig
from edsvfh.pipeline import EventDrivenVerifierPipeline
from edsvfh.train_public import build_feature_dataset, train_from_robomimic
from edsvfh.types import Episode, EpisodeStep, StepObservation


def _precomputed_episode(dim: int = 16) -> Episode:
    steps = []
    for t in range(3):
        obs = StepObservation(
            image=None,
            robot_state=np.zeros(10, dtype=np.float32),
            action=np.zeros(4, dtype=np.float32),
            policy_stats=np.zeros(4, dtype=np.float32),
            action_type='move',
            timestamp=t,
            instruction='test',
            precomputed_vector=np.full((dim,), float(t), dtype=np.float32),
            precomputed_visual_embedding=np.full((max(1, dim // 2),), float(t), dtype=np.float32),
            precomputed_action_one_hot=np.array([1, 0, 0, 0, 0, 0], dtype=np.float32),
            precomputed_object_gripper_dist=0.1,
            precomputed_object_target_dist=0.2,
            precomputed_object_height=0.3,
            precomputed_visibility=1.0,
        )
        steps.append(EpisodeStep(observation=obs))
    return Episode(task='test', instruction='test', steps=steps, outcome='failure', failure_onset=2)


def test_build_feature_dataset_uses_precomputed_without_hf(monkeypatch) -> None:
    def should_not_build_encoder(*args, **kwargs):
        raise AssertionError('build_encoder should not be called for precomputed-only inputs')

    monkeypatch.setattr('edsvfh.train_public.build_encoder', should_not_build_encoder)
    dataset = build_feature_dataset([_precomputed_episode()], AppConfig())
    assert dataset.X.shape[0] == 3


def test_pipeline_uses_precomputed_without_hf(monkeypatch, tmp_path: Path) -> None:
    fixture = tmp_path / 'tiny.hdf5'
    from edsvfh.public_data import create_tiny_robomimic_fixture
    create_tiny_robomimic_fixture(fixture)
    bundle, _metrics, _episodes = train_from_robomimic(str(fixture), config=AppConfig())

    def should_not_build_encoder(*args, **kwargs):
        raise AssertionError('pipeline should not build an HF encoder for precomputed-only inputs')

    monkeypatch.setattr('edsvfh.pipeline.build_encoder', should_not_build_encoder)
    pipeline = EventDrivenVerifierPipeline(bundle=bundle, config=AppConfig())
    out = pipeline.step(_precomputed_episode(dim=bundle.input_dim).steps[0].observation)
    assert out.timestamp == 0
