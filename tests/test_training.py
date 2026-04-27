from __future__ import annotations

from edsvfh.public_data import load_robomimic_hdf5
from edsvfh.train_public import train_from_robomimic


def test_fixture_training_metrics(trained_fixture_bundle):
    metrics = trained_fixture_bundle['metrics']
    assert metrics['subgoal_accuracy'] >= 0.70
    assert metrics['done_accuracy'] >= 0.95
    assert metrics['mean_horizon_auc'] >= 0.90


def test_reloading_and_training_from_fixture_path(trained_fixture_bundle):
    fixture_path = trained_fixture_bundle['fixture_path']
    bundle, metrics, episodes = train_from_robomimic(fixture_path, output_path=None, config=trained_fixture_bundle['config'])
    assert len(episodes) >= 4
    assert metrics['subgoal_accuracy'] >= 0.70
    assert bundle.describe()['input_dim'] > 0
