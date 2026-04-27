from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from edsvfh.config import AppConfig
from edsvfh.droid_convert import create_mock_droid_shards
from edsvfh.fino_convert import create_mock_failure_manifest_dataset
from edsvfh.fino_finetune import fine_tune_bundle_on_failure_shards
from edsvfh.sharded_train import train_bundle_from_shards
from edsvfh.train_public import train_on_fixture


@pytest.fixture(scope='session')
def trained_fixture_bundle(tmp_path_factory: pytest.TempPathFactory):
    tmp_dir = tmp_path_factory.mktemp('edsvfh_public')
    bundle_path = tmp_dir / 'bundle.pkl'
    fixture_path = tmp_dir / 'tiny_robomimic_fixture.hdf5'
    config = AppConfig()
    config.encoder.name = 'fallback'
    bundle, metrics, fixture = train_on_fixture(output_path=bundle_path, fixture_path=fixture_path, config=config)
    return {
        'bundle': bundle,
        'metrics': metrics,
        'bundle_path': bundle_path,
        'fixture_path': fixture,
        'config': config,
    }



@pytest.fixture(scope='session')
def trained_mock_droid_bundle(tmp_path_factory: pytest.TempPathFactory):
    tmp_dir = tmp_path_factory.mktemp('edsvfh_droid_mock')
    shard_dir = tmp_dir / 'mock_droid_shards'
    manifest = create_mock_droid_shards(shard_dir, num_episodes=20, steps_per_episode=20, episodes_per_shard=4, include_failures=True, seed=9)
    bundle_path = tmp_dir / 'mock_droid_bundle.pkl'
    config = AppConfig()
    config.encoder.name = 'fallback'
    result = train_bundle_from_shards(shard_dir, output_path=bundle_path, config=config, epochs=1)
    return {
        'manifest': manifest,
        'bundle': result.bundle,
        'metrics': result.metrics,
        'bundle_path': bundle_path,
        'shard_dir': shard_dir,
        'config': config,
    }


@pytest.fixture(scope="session")
def mock_failure_shards(tmp_path_factory: pytest.TempPathFactory):
    tmp_dir = tmp_path_factory.mktemp('edsvfh_failure_mock')
    root_dir = tmp_dir / 'mock_failure_manifest'
    shard_dir = tmp_dir / 'mock_failure_shards'
    manifest = create_mock_failure_manifest_dataset(root_dir, shard_dir, num_episodes=20, episodes_per_shard=4, image_size=96, seed=17)
    return {
        'root_dir': root_dir,
        'shard_dir': shard_dir,
        'manifest': manifest,
    }


@pytest.fixture(scope="session")
def fino_finetuned_bundle(trained_mock_droid_bundle, mock_failure_shards, tmp_path_factory: pytest.TempPathFactory):
    tmp_dir = tmp_path_factory.mktemp('edsvfh_failure_finetune')
    out_path = tmp_dir / 'fino_finetuned_bundle.pkl'
    config = AppConfig()
    config.encoder.name = 'fallback'
    result = fine_tune_bundle_on_failure_shards(
        trained_mock_droid_bundle['bundle'],
        mock_failure_shards['shard_dir'],
        output_path=out_path,
        config=config,
        epochs=2,
    )
    return {
        'bundle': result.bundle,
        'metrics': result.metrics,
        'bundle_path': out_path,
        'shard_dir': mock_failure_shards['shard_dir'],
        'strategies': result.strategies,
        'config': config,
    }
