from __future__ import annotations

from edsvfh.pipeline import EventDrivenVerifierPipeline
from edsvfh.public_data import load_robomimic_hdf5


def test_success_episode_remains_active(trained_fixture_bundle):
    bundle = trained_fixture_bundle['bundle']
    pipeline = EventDrivenVerifierPipeline(bundle=bundle, config=trained_fixture_bundle['config'])
    episodes = load_robomimic_hdf5(trained_fixture_bundle['fixture_path'])
    success_episode = next(ep for ep in episodes if ep.outcome == 'success')
    summary = pipeline.run_episode(success_episode)
    assert summary['terminated'] is False
    assert summary['terminal_decision'] is None


def test_drift_failure_episode_shields_before_onset(trained_fixture_bundle):
    bundle = trained_fixture_bundle['bundle']
    pipeline = EventDrivenVerifierPipeline(bundle=bundle, config=trained_fixture_bundle['config'])
    episodes = load_robomimic_hdf5(trained_fixture_bundle['fixture_path'])
    drift_episode = next(ep for ep in episodes if ep.outcome == 'failure' and ep.failure_onset == 13)
    summary = pipeline.run_episode(drift_episode)
    assert summary['terminated'] is True
    assert summary['terminal_decision'] == 'shield'
    assert summary['terminated_at'] is not None
    assert summary['terminated_at'] <= drift_episode.failure_onset
    assert summary['lead_time_to_failure'] == drift_episode.failure_onset - summary['terminated_at']


def test_post_termination_steps_require_reset(trained_fixture_bundle):
    bundle = trained_fixture_bundle['bundle']
    pipeline = EventDrivenVerifierPipeline(bundle=bundle, config=trained_fixture_bundle['config'])
    episodes = load_robomimic_hdf5(trained_fixture_bundle['fixture_path'])
    drift_episode = next(ep for ep in episodes if ep.outcome == 'failure' and ep.failure_onset == 13)
    summary = pipeline.run_episode(drift_episode)
    assert summary['terminated'] is True

    later_step = drift_episode.steps[-1].observation
    out = pipeline.step(later_step)
    assert out.post_termination is True
    assert out.terminated is True
    assert out.decision == 'terminated'

    pipeline.reset()
    recovered = pipeline.step(drift_episode.steps[0].observation)
    assert recovered.post_termination is False
