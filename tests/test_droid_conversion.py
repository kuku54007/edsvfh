from __future__ import annotations

import json

from edsvfh.droid_convert import DroidPreparedTFDSSource
from edsvfh.public_data import list_hdf5_shards, load_robomimic_hdf5


def test_mock_droid_conversion_writes_manifest_and_shards(trained_mock_droid_bundle):
    shard_dir = trained_mock_droid_bundle['shard_dir']
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
    first = episodes[0]
    assert first.source.endswith('.hdf5')
    assert first.instruction
    assert len(first.steps[0].observation.robot_state) >= 10



class _FakeIterableDataset:
    def __init__(self, items):
        self._items = items

    def __iter__(self):
        return iter(self._items)


def test_steps_to_list_accepts_iterable_dataset_like_steps():
    steps = _FakeIterableDataset([
        {
            'reward': 0.0,
            'is_last': False,
            'action': [0.1, 0.2, 0.3],
            'observation': {
                'cartesian_position': [0.0, 0.1, 0.2, 0.0, 0.0, 0.0],
                'gripper_position': [0.5],
                'joint_position': [0.0] * 7,
            },
        },
        {
            'reward': 1.0,
            'is_last': True,
            'action': [0.4, 0.5, 0.6],
            'observation': {
                'cartesian_position': [0.3, 0.4, 0.5, 0.0, 0.0, 0.0],
                'gripper_position': [0.9],
                'joint_position': [0.1] * 7,
            },
        },
    ])

    out = DroidPreparedTFDSSource._steps_to_list(steps, tfds=None)

    assert len(out) == 2
    assert float(out[0]['reward']) == 0.0
    assert out[0]['observation']['cartesian_position'].shape[0] == 6
    assert bool(out[1]['is_last']) is True


class _BatchCountingEncoder:
    def __init__(self):
        self.config = type('Cfg', (), {'convert_batch_size': 4})()
        self.calls = []

    def extract_batch(self, observations):
        self.calls.append(len(observations))
        out = []
        import numpy as np
        from edsvfh.types import FeatureSnapshot
        for obs in observations:
            vis = np.ones((8,), dtype=np.float32)
            one_hot = np.zeros((len(obs.action) if len(obs.action) else 1,), dtype=np.float32)
            vec = np.concatenate([vis, obs.robot_state.astype(np.float32), obs.action.astype(np.float32), obs.policy_stats.astype(np.float32)], dtype=np.float32)
            out.append(FeatureSnapshot(vector=vec, visual_embedding=vis, object_gripper_dist=0.0, object_target_dist=0.0, object_height=0.0, visibility=1.0, action_one_hot=np.zeros(6, dtype=np.float32)))
        return out


def test_attach_precomputed_features_uses_batched_encoder():
    import numpy as np
    from edsvfh.droid_convert import _attach_precomputed_features

    episode = {
        'images': [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(10)],
        'gripper': [np.array([0.5], dtype=np.float32) for _ in range(10)],
        'actions': [np.zeros((7,), dtype=np.float32) for _ in range(10)],
        'eef': [np.zeros((6,), dtype=np.float32) for _ in range(10)],
        'joint': [np.zeros((7,), dtype=np.float32) for _ in range(10)],
        'uncertainty': [np.array([0.0], dtype=np.float32) for _ in range(10)],
        'instruction': 'test',
    }
    enc = _BatchCountingEncoder()
    out = _attach_precomputed_features(episode, enc)
    assert out['precomputed_vector'].shape[0] == 10
    assert enc.calls == [4, 4, 2]
