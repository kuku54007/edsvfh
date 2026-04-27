from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from edsvfh.config import AppConfig
from edsvfh.fino_convert import convert_failure_manifest_to_shards
from edsvfh.encoders import build_encoder
from edsvfh.types import ACTION_TYPES, Episode, EpisodeStep, FeatureSnapshot, StepObservation
from edsvfh.fiper_pseudo_onset import (
    FIPERStyleNormalBaseline,
    _episode_feature_matrix,
    fit_droid_success_baseline,
    infer_pseudo_onset_for_episode,
    rebuild_fino_with_pseudo_onset,
    relabel_fino_manifest_with_pseudo_onsets,
)
from edsvfh.public_data import list_hdf5_shards


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def test_fit_success_baseline_and_relabel_manifest(trained_mock_droid_bundle, mock_failure_shards, tmp_path: Path) -> None:
    config = AppConfig()
    config.encoder.name = 'fallback'
    baseline_path = tmp_path / 'droid_success_baseline.pkl'
    baseline = fit_droid_success_baseline(
        trained_mock_droid_bundle['shard_dir'],
        output_path=baseline_path,
        config=config,
        feature_source='visual',
        window=3,
        phase_bins=8,
        quantile=0.95,
        show_progress=False,
    )
    assert baseline_path.exists()
    loaded = FIPERStyleNormalBaseline.load(baseline_path)
    assert loaded.describe()['num_success_episodes'] > 0
    assert loaded.describe()['feature_dim'] > 0

    pseudo_manifest = tmp_path / 'mock_fino_manifest_pseudo.jsonl'
    manifest_path = mock_failure_shards['root_dir'] / 'mock_fino_manifest.jsonl'
    result = relabel_fino_manifest_with_pseudo_onsets(
        manifest_path,
        baseline_path,
        pseudo_manifest,
        image_size=96,
        config=config,
        show_progress=False,
    )
    assert pseudo_manifest.exists()
    assert result.total_episodes > 0
    rows = _read_jsonl(pseudo_manifest)
    failures = [row for row in rows if row['outcome'] == 'failure']
    successes = [row for row in rows if row['outcome'] == 'success']
    assert failures
    assert successes
    assert all(row.get('pseudo_failure_onset') is not None for row in failures)
    assert all(row.get('failure_onset') == row.get('pseudo_failure_onset') for row in failures)
    assert all(row.get('failure_onset') in (None, '', -1) for row in successes)

    diffs: list[int] = []
    for row in failures:
        assert row.get('original_failure_onset') is not None
        num_frames = len(sorted(Path(row['frames_dir']).glob('*.png')))
        onset = int(row['pseudo_failure_onset'])
        assert 0 <= onset < num_frames
        diffs.append(abs(onset - int(row['original_failure_onset'])))
    assert float(np.mean(diffs)) <= 6.0


def test_convert_fino_manifest_can_prefer_pseudo_onset(mock_failure_shards, tmp_path: Path) -> None:
    rows = _read_jsonl(mock_failure_shards['root_dir'] / 'mock_fino_manifest.jsonl')
    failure_row = next(row for row in rows if row['outcome'] == 'failure')
    failure_row['split'] = 'train'
    failure_row['original_failure_onset'] = int(failure_row['failure_onset'])
    failure_row['pseudo_failure_onset'] = int(failure_row['failure_onset']) + 1
    failure_row['pseudo_onset_reason'] = 'unit_test'
    failure_row['pseudo_onset_confidence'] = 1.234
    manifest_path = tmp_path / 'single_pseudo_manifest.jsonl'
    manifest_path.write_text(json.dumps(failure_row) + '\n', encoding='utf-8')

    out_dir = tmp_path / 'converted_pseudo'
    manifest = convert_failure_manifest_to_shards(
        manifest_path,
        out_dir,
        episodes_per_shard=1,
        image_size=96,
        show_progress=False,
        prefer_pseudo_onset=True,
    )
    shards = manifest.train_shards or manifest.calib_shards or manifest.eval_shards
    assert shards
    with h5py.File(shards[0], 'r') as f:
        demo = f['data'][next(iter(f['data'].keys()))]
        assert int(demo.attrs['failure_onset']) == int(failure_row['pseudo_failure_onset'])
        assert int(demo.attrs['original_failure_onset']) == int(failure_row['original_failure_onset'])
        assert int(demo.attrs['pseudo_failure_onset']) == int(failure_row['pseudo_failure_onset'])
        assert str(demo.attrs['pseudo_onset_reason']) == 'unit_test'
        assert float(demo.attrs['pseudo_onset_confidence']) == 1.234


def test_rebuild_fino_pseudo_onset_runs_end_to_end(trained_mock_droid_bundle, mock_failure_shards, tmp_path: Path) -> None:
    config = AppConfig()
    config.encoder.name = 'fallback'
    result = rebuild_fino_with_pseudo_onset(
        trained_mock_droid_bundle['shard_dir'],
        mock_failure_shards['root_dir'] / 'mock_fino_manifest.jsonl',
        trained_mock_droid_bundle['bundle_path'],
        baseline_output_path=tmp_path / 'baseline.pkl',
        pseudo_manifest_output_path=tmp_path / 'pseudo_manifest.jsonl',
        converted_output_dir=tmp_path / 'fino_pseudo_shards',
        output_bundle_path=tmp_path / 'droid_fino_pseudo.pkl',
        config=config,
        epochs=1,
        feature_source='visual',
        window=3,
        phase_bins=8,
        quantile=0.95,
        image_size=96,
        show_progress=False,
    )
    assert Path(result.baseline_path).exists()
    assert Path(result.pseudo_manifest_path).exists()
    assert Path(result.output_bundle).exists()
    assert 'done_accuracy' in result.metrics
    assert len(result.strategies) > 0
    assert list_hdf5_shards(Path(result.converted_root), split='train')


def test_convert_fino_manifest_rebuilds_when_manifest_changes(mock_failure_shards, tmp_path: Path) -> None:
    rows = _read_jsonl(mock_failure_shards['root_dir'] / 'mock_fino_manifest.jsonl')
    failure_row = next(row for row in rows if row['outcome'] == 'failure')
    failure_row['split'] = 'train'
    failure_row['original_failure_onset'] = int(failure_row['failure_onset'])
    failure_row['pseudo_failure_onset'] = int(failure_row['failure_onset']) + 1
    manifest_path = tmp_path / 'single_pseudo_manifest.jsonl'
    manifest_path.write_text(json.dumps(failure_row) + "\n", encoding='utf-8')

    out_dir = tmp_path / 'converted_manifest_change'
    first = convert_failure_manifest_to_shards(
        manifest_path,
        out_dir,
        episodes_per_shard=1,
        image_size=96,
        show_progress=False,
        prefer_pseudo_onset=True,
    )
    assert first.resume_signature is not None

    failure_row['pseudo_failure_onset'] = int(failure_row['failure_onset']) + 4
    manifest_path.write_text(json.dumps(failure_row) + "\n", encoding='utf-8')
    second = convert_failure_manifest_to_shards(
        manifest_path,
        out_dir,
        episodes_per_shard=1,
        image_size=96,
        show_progress=False,
        prefer_pseudo_onset=True,
        resume=True,
    )
    assert second.resume_signature is not None
    assert second.manifest_sha256 != first.manifest_sha256

    shards = second.train_shards or second.calib_shards or second.eval_shards
    assert shards
    with h5py.File(shards[0], 'r') as f:
        demo = f['data'][next(iter(f['data'].keys()))]
        assert int(demo.attrs['failure_onset']) == int(failure_row['pseudo_failure_onset'])


def test_pseudo_onset_keeps_original_when_low_confidence(tmp_path: Path) -> None:
    config = AppConfig()
    config.encoder.name = 'fallback'
    encoder = build_encoder(config.encoder)

    image = np.zeros((96, 96, 3), dtype=np.uint8)
    steps = [
        EpisodeStep(
            observation=StepObservation(
                image=image.copy(),
                robot_state=np.zeros((10,), dtype=np.float32),
                action=np.zeros((4,), dtype=np.float32),
                policy_stats=np.zeros((4,), dtype=np.float32),
                action_type='move',
                timestamp=i,
                instruction='unit test',
            )
        )
        for i in range(8)
    ]
    episode = Episode(
        task='fino_failure',
        instruction='unit test',
        steps=steps,
        outcome='failure',
        failure_onset=5,
        source='unit-test',
    )
    features, uncertainty, _ = _episode_feature_matrix(episode, encoder, feature_source='visual')
    assert float(np.max(uncertainty)) == 0.0

    feat_dim = features.shape[1]
    phase_bins = 4
    baseline = FIPERStyleNormalBaseline(
        encoder='fallback',
        feature_source='visual',
        phase_feature_mean=np.tile(features[0:1], (phase_bins, 1)).astype(np.float32),
        phase_feature_std=np.ones((phase_bins, feat_dim), dtype=np.float32),
        phase_feature_count=np.full((phase_bins,), 100, dtype=np.int64),
        global_feature_mean=features[0].astype(np.float32),
        global_feature_std=np.ones((feat_dim,), dtype=np.float32),
        obs_thresholds=np.full((phase_bins,), 1.0, dtype=np.float32),
        uncertainty_thresholds=np.full((phase_bins,), 0.5, dtype=np.float32),
        global_obs_threshold=1.0,
        global_uncertainty_threshold=0.5,
        window=3,
        phase_bins=phase_bins,
        quantile=0.95,
        min_phase_count=1,
        num_success_episodes=1,
        num_success_steps=features.shape[0],
        metadata={'source': 'unit-test'},
    )

    result = infer_pseudo_onset_for_episode(episode, baseline, config=config, encoder=encoder)
    assert result.pseudo_failure_onset == 5
    assert result.reason == 'kept_original_low_confidence'
    assert result.confidence == 0.0


def test_pseudo_onset_auto_resolves_encoder_dimension_mismatch(monkeypatch) -> None:
    from edsvfh import fiper_pseudo_onset as fpo

    config = AppConfig()
    config.encoder.name = 'fallback'
    fallback_encoder = build_encoder(config.encoder)

    image = np.zeros((96, 96, 3), dtype=np.uint8)
    steps = [
        EpisodeStep(
            observation=StepObservation(
                image=image.copy(),
                robot_state=np.zeros((10,), dtype=np.float32),
                action=np.zeros((4,), dtype=np.float32),
                policy_stats=np.zeros((4,), dtype=np.float32),
                action_type='move',
                timestamp=i,
                instruction='unit test',
            )
        )
        for i in range(6)
    ]
    episode = Episode(
        task='fino_failure',
        instruction='unit test',
        steps=steps,
        outcome='failure',
        failure_onset=4,
        source='unit-test',
    )

    expected_dim = 1549
    phase_bins = 4
    baseline = FIPERStyleNormalBaseline(
        encoder='fallback',
        feature_source='visual',
        phase_feature_mean=np.zeros((phase_bins, expected_dim), dtype=np.float32),
        phase_feature_std=np.ones((phase_bins, expected_dim), dtype=np.float32),
        phase_feature_count=np.full((phase_bins,), 100, dtype=np.int64),
        global_feature_mean=np.zeros((expected_dim,), dtype=np.float32),
        global_feature_std=np.ones((expected_dim,), dtype=np.float32),
        obs_thresholds=np.full((phase_bins,), 1.0, dtype=np.float32),
        uncertainty_thresholds=np.full((phase_bins,), 0.5, dtype=np.float32),
        global_obs_threshold=1.0,
        global_uncertainty_threshold=0.5,
        window=3,
        phase_bins=phase_bins,
        quantile=0.95,
        min_phase_count=1,
        num_success_episodes=1,
        num_success_steps=len(steps),
        metadata={'source': 'unit-test'},
    )

    real_build_encoder = build_encoder

    class FakeHFEncoder:
        def __init__(self, name: str = 'siglip2_dinov2') -> None:
            self.config = SimpleNamespace(name=name)

        def extract(self, obs: StepObservation) -> FeatureSnapshot:
            return self.extract_batch([obs])[0]

        def extract_batch(self, observations):
            snaps = []
            for idx, _obs in enumerate(observations):
                visual = np.zeros((expected_dim,), dtype=np.float32)
                visual[min(idx, expected_dim - 1)] = 1.0
                snaps.append(
                    FeatureSnapshot(
                        vector=visual.copy(),
                        visual_embedding=visual,
                        object_gripper_dist=0.0,
                        object_target_dist=0.0,
                        object_height=0.0,
                        visibility=1.0,
                        action_one_hot=np.zeros((len(ACTION_TYPES),), dtype=np.float32),
                    )
                )
            return snaps

    def fake_build_encoder(cfg):
        if str(cfg.name).lower() == 'siglip2_dinov2':
            return FakeHFEncoder('siglip2_dinov2')
        return real_build_encoder(cfg)

    monkeypatch.setattr(fpo, 'build_encoder', fake_build_encoder)

    result = infer_pseudo_onset_for_episode(episode, baseline, config=config, encoder=fallback_encoder)
    assert result.pseudo_failure_onset == 4
    assert result.reason == 'kept_original_low_confidence'
    assert result.confidence <= 0.10
