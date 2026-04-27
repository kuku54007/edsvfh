from __future__ import annotations

import json
import subprocess
from pathlib import Path


from edsvfh.paper_pack import redact_mapping, summarize_manifest


def test_redact_mapping_masks_sensitive_keys() -> None:
    env = {
        'OPENAI_API_KEY': 'secret-value',
        'RUN_NAME': 'exp-01',
        'PASSWORD_HINT': 'abc',
        'PROJECT_DIR': '/workspace/project',
    }
    redacted = redact_mapping(env)
    assert redacted['OPENAI_API_KEY'] == '<REDACTED>'
    assert redacted['PASSWORD_HINT'] == '<REDACTED>'
    assert redacted['RUN_NAME'] == 'exp-01'
    assert redacted['PROJECT_DIR'] == '/workspace/project'


def test_summarize_manifest_counts_and_shift(tmp_path) -> None:
    manifest = tmp_path / 'fino_manifest_pseudo.jsonl'
    rows = [
        {
            'episode_id': 'ep-1',
            'outcome': 'failure',
            'failure_onset': 5,
            'original_failure_onset': 7,
            'pseudo_failure_onset': 5,
            'pseudo_onset_reason': 'dual_signal_threshold',
            'pseudo_onset_confidence': 0.82,
        },
        {
            'episode_id': 'ep-2',
            'outcome': 'success',
            'failure_onset': None,
        },
        {
            'episode_id': 'ep-3',
            'outcome': 'failure',
            'failure_onset': 10,
            'original_failure_onset': 10,
            'pseudo_failure_onset': 10,
            'pseudo_onset_reason': 'kept_original',
            'pseudo_onset_confidence': 0.61,
        },
    ]
    manifest.write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')

    summary = summarize_manifest(manifest, topk_changed=5)
    assert summary is not None
    assert summary['episodes'] == 3
    assert summary['outcome_counts'] == {'failure': 2, 'success': 1}
    assert summary['pseudo_failure_onset_count'] == 2
    assert summary['original_failure_onset_count'] == 2
    assert summary['replaced_failure_onset_count'] == 2
    assert summary['unchanged_onset_count'] == 1
    assert summary['pseudo_onset_reason_counts']['dual_signal_threshold'] == 1
    assert summary['pseudo_onset_reason_counts']['kept_original'] == 1
    assert summary['onset_shift']['min'] == -2
    assert summary['onset_shift']['max'] == 0


def test_paper_pack_config_ignores_stale_project_dir_by_default(monkeypatch) -> None:
    from edsvfh.paper_pack import PaperPackConfig

    monkeypatch.setenv('PROJECT_DIR', '/workspace/stale_project_dir')
    config = PaperPackConfig.from_env()
    expected = Path(__file__).resolve().parents[1]
    assert config.project_dir == expected


def test_runpod_common_prefers_script_project_dir_over_stale_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    common_sh = repo_root / 'scripts' / 'runpod' / 'common.sh'
    cmd = [
        'bash',
        '-lc',
        (
            f'export PROJECT_DIR="/workspace/stale_project_dir"; '
            f'unset PYTHONPATH; '
            f'source "{common_sh}"; '
            f'printf "%s\n%s\n" "$PROJECT_DIR" "$PYTHONPATH"'
        ),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = proc.stdout.strip().splitlines()
    assert lines[0] == str(repo_root)
    assert lines[1].split(':')[0] == str(repo_root)
