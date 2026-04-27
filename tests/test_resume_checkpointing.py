from __future__ import annotations

from pathlib import Path

from edsvfh.config import AppConfig
from edsvfh.droid_convert import MockDroidEpisodeSource, convert_droid_source_to_shards
from edsvfh.fino_convert import create_mock_failure_manifest_dataset
from edsvfh.fino_finetune import fine_tune_bundle_on_failure_shards
from edsvfh.sharded_train import train_bundle_from_shards


def test_convert_droid_resume_from_partial(tmp_path: Path) -> None:
    out = tmp_path / 'droid_resume'
    source_partial = MockDroidEpisodeSource(num_episodes=4, steps_per_episode=8, seed=1)
    manifest_partial = convert_droid_source_to_shards(
        source_partial,
        out,
        source_label='mock',
        source_mode='mock',
        episodes_per_shard=2,
        checkpoint_every=1,
        resume=True,
    )
    assert manifest_partial.train_episodes + manifest_partial.calib_episodes + manifest_partial.eval_episodes == 4

    source_full = MockDroidEpisodeSource(num_episodes=8, steps_per_episode=8, seed=1)
    manifest_full = convert_droid_source_to_shards(
        source_full,
        out,
        source_label='mock',
        source_mode='mock',
        episodes_per_shard=2,
        checkpoint_every=1,
        resume=True,
    )
    assert manifest_full.train_episodes + manifest_full.calib_episodes + manifest_full.eval_episodes == 8
    assert (out / '.convert_state.json').exists()


def test_train_and_fino_resume_completed_checkpoint(tmp_path: Path) -> None:
    shards = tmp_path / 'mock_shards'
    convert_droid_source_to_shards(
        MockDroidEpisodeSource(num_episodes=10, steps_per_episode=10, seed=2),
        shards,
        source_label='mock',
        source_mode='mock',
        episodes_per_shard=3,
        checkpoint_every=1,
        resume=True,
    )
    bundle_path = tmp_path / 'bundle.pkl'
    train_res = train_bundle_from_shards(
        shards,
        output_path=bundle_path,
        config=AppConfig(),
        epochs=1,
        checkpoint_every_shards=1,
        resume=True,
    )
    assert Path(train_res.checkpoint_path).exists()
    train_res_2 = train_bundle_from_shards(
        shards,
        output_path=bundle_path,
        config=AppConfig(),
        epochs=1,
        checkpoint_every_shards=1,
        resume=True,
    )
    assert train_res_2.metrics.keys() == train_res.metrics.keys()

    failure_root = tmp_path / 'failroot'
    failure_shards = tmp_path / 'failshards'
    create_mock_failure_manifest_dataset(failure_root, failure_shards, num_episodes=6, episodes_per_shard=2, seed=3)
    fino_path = tmp_path / 'fino.pkl'
    ft_res = fine_tune_bundle_on_failure_shards(
        bundle_path,
        failure_shards,
        output_path=fino_path,
        config=AppConfig(),
        epochs=1,
        checkpoint_every_shards=1,
        resume=True,
    )
    assert Path(ft_res.checkpoint_path).exists()
    ft_res_2 = fine_tune_bundle_on_failure_shards(
        bundle_path,
        failure_shards,
        output_path=fino_path,
        config=AppConfig(),
        epochs=1,
        checkpoint_every_shards=1,
        resume=True,
    )
    assert ft_res_2.metrics.keys() == ft_res.metrics.keys()

def test_convert_droid_with_precomputed_features(tmp_path: Path) -> None:
    out = tmp_path / 'droid_precomp'
    manifest = convert_droid_source_to_shards(
        MockDroidEpisodeSource(num_episodes=4, steps_per_episode=6, seed=4),
        out,
        source_label='mock',
        source_mode='mock',
        episodes_per_shard=2,
        checkpoint_every=1,
        precompute_encoder='fallback',
        resume=True,
    )
    assert manifest.train_shards or manifest.calib_shards or manifest.eval_shards
    import h5py
    shard = Path((manifest.train_shards or manifest.calib_shards or manifest.eval_shards)[0])
    with h5py.File(shard, 'r') as f:
        demo = f['data'][sorted(f['data'].keys())[0]]
        assert 'precomputed_vector' in demo['obs']
        assert 'precomputed_visual_embedding' in demo['obs']

def test_train_from_precomputed_droid_shards(tmp_path: Path) -> None:
    out = tmp_path / 'droid_precomp_train'
    convert_droid_source_to_shards(
        MockDroidEpisodeSource(num_episodes=6, steps_per_episode=6, seed=6),
        out,
        source_label='mock',
        source_mode='mock',
        episodes_per_shard=2,
        checkpoint_every=1,
        precompute_encoder='fallback',
        resume=True,
    )
    bundle_path = tmp_path / 'precomp_bundle.pkl'
    res = train_bundle_from_shards(out, output_path=bundle_path, config=AppConfig(), epochs=1, checkpoint_every_shards=1, resume=True)
    assert bundle_path.exists()
    assert 'subgoal_accuracy' in res.metrics


def test_convert_droid_resume_quarantines_corrupt_trailing_shard(tmp_path: Path) -> None:
    out = tmp_path / 'droid_resume_corrupt'
    convert_droid_source_to_shards(
        MockDroidEpisodeSource(num_episodes=5, steps_per_episode=8, seed=7),
        out,
        source_label='mock',
        source_mode='mock',
        episodes_per_shard=2,
        checkpoint_every=1,
        resume=True,
    )

    train_shards = sorted((out / 'train').glob('*.hdf5'))
    assert train_shards
    last = train_shards[-1]
    last.write_bytes(b'corrupt-hdf5')

    manifest = convert_droid_source_to_shards(
        MockDroidEpisodeSource(num_episodes=8, steps_per_episode=8, seed=7),
        out,
        source_label='mock',
        source_mode='mock',
        episodes_per_shard=2,
        checkpoint_every=1,
        resume=True,
    )
    total = manifest.train_episodes + manifest.calib_episodes + manifest.eval_episodes
    assert total == 8
    quarantined = list((out / 'train').glob('*.corrupt.*'))
    assert quarantined


def test_convert_droid_resume_quarantines_runtimeerror_trailing_shard(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / 'droid_resume_runtimeerr'
    convert_droid_source_to_shards(
        MockDroidEpisodeSource(num_episodes=5, steps_per_episode=8, seed=8),
        out,
        source_label='mock',
        source_mode='mock',
        episodes_per_shard=2,
        checkpoint_every=1,
        resume=True,
    )

    train_shards = sorted((out / 'train').glob('*.hdf5'))
    assert train_shards
    last = train_shards[-1]

    from edsvfh.droid_convert import _RobomimicShardSink
    orig = _RobomimicShardSink._count_episodes_in_file

    def flaky(self, path):
        if path == last:
            raise RuntimeError('simulated group corruption')
        return orig(self, path)

    monkeypatch.setattr(_RobomimicShardSink, '_count_episodes_in_file', flaky)

    manifest = convert_droid_source_to_shards(
        MockDroidEpisodeSource(num_episodes=8, steps_per_episode=8, seed=8),
        out,
        source_label='mock',
        source_mode='mock',
        episodes_per_shard=2,
        checkpoint_every=1,
        resume=True,
    )
    total = manifest.train_episodes + manifest.calib_episodes + manifest.eval_episodes
    assert total == 8
    quarantined = list((out / 'train').glob('*.corrupt.*'))
    assert quarantined
