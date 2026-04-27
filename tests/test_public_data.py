from __future__ import annotations

from pathlib import Path

from edsvfh.public_data import dataset_catalog, load_lerobot_info_json, load_robomimic_hdf5


def test_dataset_catalog_contains_expected_entries():
    names = {item['name'] for item in dataset_catalog()}
    assert {'DROID', 'BridgeData V2', 'LIBERO', 'robomimic'} <= names


def test_lerobot_metadata_fixture_parses():
    info = load_lerobot_info_json(Path(__file__).parent / 'fixtures' / 'lerobot_pusht_info.json')
    assert info['codebase_version'] == 'v3.0'
    assert info['total_episodes'] == 206
    assert info['fps'] == 10


def test_fixture_loader_reads_episodes(trained_fixture_bundle):
    episodes = load_robomimic_hdf5(trained_fixture_bundle['fixture_path'])
    assert len(episodes) >= 20
    assert episodes[0].steps[0].observation.image is not None
