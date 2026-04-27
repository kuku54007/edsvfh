from __future__ import annotations

from edsvfh.pipeline import EventDrivenVerifierPipeline
from edsvfh.public_data import list_hdf5_shards, load_robomimic_hdf5


def test_sharded_training_produces_bundle_and_metrics(trained_mock_droid_bundle):
    metrics = trained_mock_droid_bundle['metrics']
    assert trained_mock_droid_bundle['bundle'].describe()['training_mode'] == 'sharded_incremental'
    assert metrics['subgoal_accuracy'] >= 0.55
    assert metrics['done_accuracy'] >= 0.80


def test_sharded_bundle_runs_online_pipeline(trained_mock_droid_bundle):
    bundle = trained_mock_droid_bundle['bundle']
    config = trained_mock_droid_bundle['config']
    shard_dir = trained_mock_droid_bundle['shard_dir']
    eval_shards = list_hdf5_shards(shard_dir, split='eval')
    episodes = load_robomimic_hdf5(eval_shards[0])
    pipeline = EventDrivenVerifierPipeline(bundle=bundle, config=config)
    summary = pipeline.run_episode(episodes[0])
    assert 'events' in summary
    assert summary['last_processed_timestamp'] is not None
