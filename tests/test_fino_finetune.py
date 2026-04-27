from __future__ import annotations

import json

from edsvfh.pipeline import EventDrivenVerifierPipeline
from edsvfh.public_data import list_hdf5_shards, load_robomimic_hdf5


def test_mock_failure_manifest_conversion_writes_shards(mock_failure_shards):
    shard_dir = mock_failure_shards['shard_dir']
    manifest_path = shard_dir / 'manifest.json'
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['train_episodes'] > 0
    assert manifest['calib_episodes'] > 0
    assert manifest['eval_episodes'] > 0
    train_shards = list_hdf5_shards(shard_dir, split='train')
    assert train_shards
    episodes = load_robomimic_hdf5(train_shards[0])
    assert episodes
    assert episodes[0].source.endswith('.hdf5')


def test_fino_finetune_produces_bundle_and_metrics(fino_finetuned_bundle):
    bundle = fino_finetuned_bundle['bundle']
    metrics = fino_finetuned_bundle['metrics']
    desc = bundle.describe()
    assert 'failure_finetune_source' in desc
    assert desc['failure_finetune_epochs'] == 2
    assert 'done_accuracy' in metrics
    assert metrics['mean_horizon_auc'] >= 0.50
    assert len(fino_finetuned_bundle['strategies']) == len(bundle.horizons)


def test_fino_finetuned_bundle_runs_online_pipeline(fino_finetuned_bundle):
    bundle = fino_finetuned_bundle['bundle']
    config = fino_finetuned_bundle['config']
    shard_dir = fino_finetuned_bundle['shard_dir']
    eval_shards = list_hdf5_shards(shard_dir, split='eval')
    episodes = load_robomimic_hdf5(eval_shards[0])
    pipeline = EventDrivenVerifierPipeline(bundle=bundle, config=config)
    summary = pipeline.run_episode(episodes[0])
    assert 'events' in summary
    assert summary['last_processed_timestamp'] is not None


def test_fino_finetune_accepts_larger_horizon_set(trained_mock_droid_bundle, mock_failure_shards, tmp_path):
    from edsvfh.config import AppConfig
    from edsvfh.fino_finetune import fine_tune_bundle_on_failure_shards

    config = AppConfig()
    config.encoder.name = 'fallback'
    config.training.horizons = (1, 3, 5, 10, 15)
    out_path = tmp_path / 'larger_horizons_bundle.pkl'
    result = fine_tune_bundle_on_failure_shards(
        trained_mock_droid_bundle['bundle_path'],
        mock_failure_shards['shard_dir'],
        output_path=out_path,
        config=config,
        epochs=1,
        show_progress=False,
        resume=False,
    )
    assert out_path.exists()
    assert tuple(result.bundle.horizons) == (1, 3, 5, 10, 15)
    assert len(result.bundle.horizon_models) == 5
    assert len(result.strategies) == 5
    assert 'horizon_10_auc' in result.metrics
    assert 'horizon_15_auc' in result.metrics


def test_larger_horizon_freezes_existing_heads(fino_finetuned_bundle, mock_failure_shards, tmp_path):
    import pickle
    from edsvfh.config import AppConfig
    from edsvfh.fino_finetune import fine_tune_bundle_on_failure_shards

    base_bundle = fino_finetuned_bundle['bundle']
    before_models = [pickle.dumps(m) for m in base_bundle.horizon_models]
    before_calibrators = [pickle.dumps(c) for c in base_bundle.horizon_calibrators]

    config = AppConfig()
    config.encoder.name = 'fallback'
    config.training.horizons = (1, 3, 5, 10, 15)
    out_path = tmp_path / 'larger_horizons_frozen_bundle.pkl'

    result = fine_tune_bundle_on_failure_shards(
        base_bundle,
        mock_failure_shards['shard_dir'],
        output_path=out_path,
        config=config,
        epochs=1,
        update_scaler=False,
        show_progress=False,
        resume=False,
        freeze_existing_horizons=True,
    )

    assert out_path.exists()
    assert tuple(result.bundle.horizons) == (1, 3, 5, 10, 15)
    assert result.strategies[:3] == ['kept_existing_frozen'] * 3
    assert result.strategies[3:] == ['reinitialized_sgd', 'reinitialized_sgd']
    assert result.bundle.metadata['failure_finetune_train_horizon_flags'] == [False, False, False, True, True]
    assert result.bundle.metadata['failure_finetune_freeze_existing_horizons'] is True
    for idx in range(3):
        assert pickle.dumps(result.bundle.horizon_models[idx]) == before_models[idx]
        assert pickle.dumps(result.bundle.horizon_calibrators[idx]) == before_calibrators[idx]
