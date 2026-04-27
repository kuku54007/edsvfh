from __future__ import annotations

import base64

import cv2
from fastapi.testclient import TestClient

from edsvfh.api import create_app
from edsvfh.config import AppConfig
from edsvfh.public_data import load_robomimic_hdf5


def _fallback_api_config(*, force_shield: bool = False) -> AppConfig:
    """Return a deterministic, CPU-only config for API contract tests."""
    config = AppConfig()
    config.encoder.name = 'fallback'
    config.encoder.device = 'cpu'
    if force_shield:
        # Exercise terminal-state mechanics deterministically. The tiny fixture is
        # intentionally small, so its learned risk can vary across sklearn/numpy
        # builds and should not be used as a brittle threshold oracle.
        config.watcher.trigger_threshold = 0.0
        config.watcher.heartbeat_steps = 1
        config.decision.stop_threshold = 0.0
        config.decision.warning_threshold = 0.0
        config.decision.confirm_count = 1
        config.decision.terminal_decisions = ('shield',)
    return config


def _payload_from_observation(obs):
    ok, enc = cv2.imencode('.png', obs.image)
    assert ok
    return {
        'timestamp': obs.timestamp,
        'action_type': obs.action_type,
        'robot_state': obs.robot_state.tolist(),
        'action': obs.action.tolist(),
        'policy_stats': obs.policy_stats.tolist(),
        'image_png_b64': base64.b64encode(enc.tobytes()).decode('utf-8'),
        'instruction': obs.instruction,
    }


def test_api_health_and_bundle(trained_fixture_bundle):
    app = create_app(bundle_path=trained_fixture_bundle['bundle_path'], config=_fallback_api_config())
    client = TestClient(app)
    assert client.get('/health').json()['status'] == 'ok'
    info = client.get('/v1/bundle').json()
    assert info['num_subgoals'] == 5
    status = client.get('/v1/status').json()
    assert status['terminated'] is False


def test_api_step(trained_fixture_bundle):
    app = create_app(bundle_path=trained_fixture_bundle['bundle_path'], config=_fallback_api_config())
    client = TestClient(app)
    episode = load_robomimic_hdf5(trained_fixture_bundle['fixture_path'])[0]
    resp = client.post('/v1/step', json=_payload_from_observation(episode.steps[0].observation))
    assert resp.status_code == 200
    body = resp.json()
    assert 'triggered' in body


def test_api_termination_and_reset(trained_fixture_bundle):
    app = create_app(bundle_path=trained_fixture_bundle['bundle_path'], config=_fallback_api_config(force_shield=True))
    client = TestClient(app)
    drift_episode = next(ep for ep in load_robomimic_hdf5(trained_fixture_bundle['fixture_path']) if ep.outcome == 'failure' and ep.failure_onset == 13)

    termination_body = None
    for step in drift_episode.steps:
        termination_body = client.post('/v1/step', json=_payload_from_observation(step.observation)).json()
        if termination_body['terminated']:
            break

    assert termination_body is not None
    assert termination_body['terminated'] is True
    assert termination_body['terminal_decision'] == 'shield'

    status = client.get('/v1/status').json()
    assert status['terminated'] is True

    later = drift_episode.steps[-1].observation
    post = client.post('/v1/step', json=_payload_from_observation(later)).json()
    assert post['post_termination'] is True

    reset_resp = client.post('/v1/reset')
    assert reset_resp.status_code == 200
    assert client.get('/v1/status').json()['terminated'] is False
