from __future__ import annotations

import json
from pathlib import Path

from edsvfh.manifest_tools import generate_fino_manifest_from_episode_dirs


def test_generate_fino_manifest_from_mock_failure_dataset(tmp_path: Path) -> None:
    episodes_root = tmp_path / 'episodes'
    for idx, suffix in enumerate(['success', 'miss_grasp', 'drift']):
        ep = episodes_root / f'ep_{idx:04d}_{suffix}'
        rgb = ep / 'rgb'
        rgb.mkdir(parents=True, exist_ok=True)
        (rgb / '000000.png').write_bytes(b'png')
        (ep / 'eef.npy').write_bytes(b'\x93NUMPY')
    manifest_path = tmp_path / 'manifest.jsonl'
    out = generate_fino_manifest_from_episode_dirs(tmp_path, manifest_path)
    assert out.exists()
    rows = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(rows) == 3
    assert rows[0]['outcome'] in {'success', 'failure'}
    assert any(r['outcome'] == 'failure' for r in rows[1:])
    assert all('frames_dir' in r for r in rows)


def test_generate_fino_manifest_from_failnet_layout(tmp_path: Path) -> None:
    root = tmp_path
    rgb_root = root / 'failnet_dataset' / 'rgb_imgs'
    # task / episode style directories
    ep_success = rgb_root / 'place' / 'ep_0001_success'
    ep_fail = rgb_root / 'place' / 'ep_0002'
    ep_success.mkdir(parents=True, exist_ok=True)
    ep_fail.mkdir(parents=True, exist_ok=True)
    (ep_success / '000000.png').write_bytes(b'png')
    (ep_fail / '000000.png').write_bytes(b'png')
    # task-specific annotation file listing a failing episode id
    (root / 'place_annotation.txt').write_text('ep_0002\n', encoding='utf-8')

    manifest_path = tmp_path / 'failnet_manifest.jsonl'
    out = generate_fino_manifest_from_episode_dirs(root, manifest_path)
    rows = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(rows) == 2
    by_id = {row['episode_id']: row for row in rows}
    assert by_id['ep_0002']['outcome'] == 'failure'
    assert by_id['ep_0001_success']['outcome'] == 'success'
    assert by_id['ep_0002']['task'] == 'place'
    assert by_id['ep_0002']['frames_dir'].endswith('failnet_dataset/rgb_imgs/place/ep_0002')
