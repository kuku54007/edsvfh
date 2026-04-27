from __future__ import annotations

from edsvfh.eval_protocols import evaluate_ablation_suite, evaluate_replay_protocol


def test_ablation_suite_outputs_rows(fino_finetuned_bundle):
    result = evaluate_ablation_suite(
        [('main', fino_finetuned_bundle['bundle_path'])],
        fino_finetuned_bundle['shard_dir'],
        variants=('calibrated_monotonic', 'raw'),
    )
    assert result['bundles'][0]['name'] == 'main'
    assert result['rows']
    assert any(row['variant'] == 'calibrated_monotonic' and row['horizon'] == 'mean' for row in result['rows'])
    assert any(row['variant'] == 'raw' and row['horizon'] == 'mean' for row in result['rows'])


def test_replay_protocol_outputs_fixed_rate_and_event_driven(fino_finetuned_bundle):
    result = evaluate_replay_protocol(
        fino_finetuned_bundle['bundle_path'],
        fino_finetuned_bundle['shard_dir'],
        fixed_rates=(1, 3),
        max_episodes=3,
    )
    strategies = {row['strategy'] for row in result['summaries']}
    assert {'event_driven', 'fixed_rate_1', 'fixed_rate_3'} <= strategies
    assert result['comparisons']
    assert result['episode_rows']
