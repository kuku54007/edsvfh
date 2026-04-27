from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, roc_auc_score

from .config import AppConfig
from .decision import DecisionLayer
from .models import VerifierBundle, adapt_feature_dim
from .pipeline import EventDrivenVerifierPipeline
from .public_data import list_hdf5_shards, load_robomimic_hdf5
from .sharded_train import _build_feature_dataset_from_shard  # internal helper, used only for offline evaluation
from .train_public import build_feature_dataset


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _predict_binary_probability(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, 'predict_proba'):
        prob = np.asarray(model.predict_proba(x), dtype=np.float64)
        if prob.ndim == 1:
            return prob.reshape(-1)
        if prob.shape[1] == 1:
            return prob[:, 0]
        return prob[:, 1]
    if hasattr(model, 'decision_function'):
        return _sigmoid(np.asarray(model.decision_function(x), dtype=np.float64))
    pred = np.asarray(model.predict(x), dtype=np.float64)
    return np.clip(pred, 0.0, 1.0)


def _as_float(value: Any) -> float | str:
    try:
        f = float(value)
    except Exception:
        return str(value)
    if math.isnan(f) or math.isinf(f):
        return 'nan' if math.isnan(f) else str(f)
    return f


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, 'item'):
        try:
            return _clean(obj.item())
        except Exception:
            pass
    if isinstance(obj, float):
        return _as_float(obj)
    return obj


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in raw.replace(';', ',').replace(' ', ',').split(','):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f'values must be positive integers, got {value}')
        values.append(value)
    return tuple(dict.fromkeys(values))


def _parse_csv_strings(raw: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in raw.replace(';', ',').split(',') if token.strip())


def _parse_bundle_spec(raw: str) -> tuple[str, Path]:
    if '=' in raw:
        name, path = raw.split('=', 1)
        return name.strip(), Path(path.strip())
    path = Path(raw)
    return path.stem, path


def _ece_binary(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    prob = np.asarray(prob, dtype=float).reshape(-1)
    if y_true.size == 0:
        return float('nan')
    prob = np.clip(prob, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(prob)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (float(np.sum(mask)) / n) * abs(acc - conf)
    return float(ece)


def _metric_row(y_true: np.ndarray, prob: np.ndarray, *, ece_bins: int = 10) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.clip(np.asarray(prob, dtype=float).reshape(-1), 0.0, 1.0)
    out: dict[str, Any] = {
        'n': int(y.size),
        'positive_count': int(np.sum(y == 1)),
        'negative_count': int(np.sum(y == 0)),
        'positive_rate': float(np.mean(y)) if y.size else float('nan'),
        'brier': float(brier_score_loss(y, p)) if y.size else float('nan'),
        'ece': _ece_binary(y, p, bins=ece_bins) if y.size else float('nan'),
    }
    if y.size == 0 or len(np.unique(y)) < 2:
        out['auc'] = float('nan')
        out['auprc'] = float('nan')
    else:
        out['auc'] = float(roc_auc_score(y, p))
        out['auprc'] = float(average_precision_score(y, p))
    return out


def _bundle_horizon_probs(bundle: VerifierBundle, x: np.ndarray, *, variant: str) -> np.ndarray:
    x = adapt_feature_dim(np.asarray(x, dtype=np.float32), bundle.input_dim)
    if bundle.feature_scaler is not None:
        xs = bundle.feature_scaler.transform(x)
    else:
        xs = x
    probs: list[np.ndarray] = []
    use_calibration = variant in {'calibrated', 'calibrated_monotonic'}
    use_monotonic = variant in {'raw_monotonic', 'calibrated_monotonic'}
    for model, calibrator in zip(bundle.horizon_models, bundle.horizon_calibrators):
        if model is None:
            base_prob = np.full((len(xs),), 0.5, dtype=np.float64)
        else:
            base_prob = _predict_binary_probability(model, xs)
        if use_calibration:
            prob = calibrator.predict(base_prob)
        else:
            prob = base_prob
        probs.append(np.asarray(prob, dtype=np.float64))
    if not probs:
        return np.zeros((len(xs), 0), dtype=np.float64)
    mat = np.stack(probs, axis=1)
    if use_monotonic:
        mat = np.maximum.accumulate(mat, axis=1)
    return mat


def evaluate_ablation_suite(
    bundle_specs: list[tuple[str, Path]],
    shard_dir: str | Path,
    *,
    variants: Iterable[str] = ('calibrated_monotonic', 'calibrated', 'raw_monotonic', 'raw'),
    ece_bins: int = 10,
) -> dict[str, Any]:
    shard_dir = Path(shard_dir)
    output: dict[str, Any] = {
        'shard_dir': str(shard_dir),
        'split': 'eval',
        'variants': list(variants),
        'bundles': [],
        'rows': [],
    }
    eval_shards = list_hdf5_shards(shard_dir, split='eval')
    if not eval_shards:
        raise FileNotFoundError(f'No eval shards found under {shard_dir / "eval"}')
    for bundle_name, bundle_path in bundle_specs:
        if not bundle_path.exists():
            output['bundles'].append({'name': bundle_name, 'path': str(bundle_path), 'missing': True})
            continue
        bundle = VerifierBundle.load(bundle_path)
        config = AppConfig()
        config.training.horizons = tuple(bundle.horizons)
        h_true_parts = [[] for _ in bundle.horizons]
        h_prob_parts = {variant: [[] for _ in bundle.horizons] for variant in variants}
        comp_true_parts: list[np.ndarray] = []
        comp_pred_parts: list[np.ndarray] = []
        done_true_parts: list[np.ndarray] = []
        done_prob_parts: list[np.ndarray] = []
        n_samples = 0
        for shard_path in eval_shards:
            dataset = _build_feature_dataset_from_shard(shard_path, config)
            X = dataset.X.astype(np.float32)
            _, completion, done, _ = bundle.predict(X)
            comp_true_parts.append(dataset.y_completion.astype(np.float32))
            comp_pred_parts.append(completion.astype(np.float32))
            done_true_parts.append(dataset.y_done.astype(int))
            done_prob_parts.append(done.astype(np.float32))
            n_samples += int(X.shape[0])
            for h_idx, _ in enumerate(bundle.horizons):
                h_true_parts[h_idx].append(dataset.y_h[:, h_idx].astype(int))
            for variant in variants:
                pred_h = _bundle_horizon_probs(bundle, X, variant=variant)
                for h_idx, _ in enumerate(bundle.horizons):
                    h_prob_parts[variant][h_idx].append(pred_h[:, h_idx].astype(np.float32))
        bundle_summary = {
            'name': bundle_name,
            'path': str(bundle_path),
            'horizons': list(bundle.horizons),
            'n_eval_shards': len(eval_shards),
            'n_eval_samples': n_samples,
        }
        if comp_true_parts:
            comp_true = np.concatenate(comp_true_parts)
            comp_pred = np.concatenate(comp_pred_parts)
            done_true = np.concatenate(done_true_parts)
            done_prob = np.concatenate(done_prob_parts)
            bundle_summary['completion_mae'] = float(mean_absolute_error(comp_true, comp_pred))
            bundle_summary['done_accuracy'] = float(np.mean((done_prob >= 0.5).astype(int) == done_true))
        output['bundles'].append(bundle_summary)
        for variant in variants:
            auc_values: list[float] = []
            auprc_values: list[float] = []
            brier_values: list[float] = []
            ece_values: list[float] = []
            for h_idx, horizon in enumerate(bundle.horizons):
                y = np.concatenate(h_true_parts[h_idx]) if h_true_parts[h_idx] else np.zeros((0,), dtype=int)
                p = np.concatenate(h_prob_parts[variant][h_idx]) if h_prob_parts[variant][h_idx] else np.zeros((0,), dtype=float)
                row = _metric_row(y, p, ece_bins=ece_bins)
                row.update({'bundle': bundle_name, 'variant': variant, 'horizon': int(horizon)})
                output['rows'].append(_clean(row))
                if isinstance(row['auc'], float) and not math.isnan(row['auc']):
                    auc_values.append(float(row['auc']))
                if isinstance(row['auprc'], float) and not math.isnan(row['auprc']):
                    auprc_values.append(float(row['auprc']))
                if isinstance(row['brier'], float) and not math.isnan(row['brier']):
                    brier_values.append(float(row['brier']))
                if isinstance(row['ece'], float) and not math.isnan(row['ece']):
                    ece_values.append(float(row['ece']))
            mean_row = {
                'bundle': bundle_name,
                'variant': variant,
                'horizon': 'mean',
                'n': n_samples,
                'positive_count': '',
                'negative_count': '',
                'positive_rate': '',
                'auc': float(np.mean(auc_values)) if auc_values else float('nan'),
                'auprc': float(np.mean(auprc_values)) if auprc_values else float('nan'),
                'brier': float(np.mean(brier_values)) if brier_values else float('nan'),
                'ece': float(np.mean(ece_values)) if ece_values else float('nan'),
            }
            output['rows'].append(_clean(mean_row))
    return _clean(output)


@dataclass
class EpisodeReplayResult:
    strategy: str
    episode_id: str
    n_steps: int
    outcome: str
    failure_onset: int | None
    call_count: int
    verifier_call_rate: float
    first_alarm_step: int | None
    first_terminal_step: int | None
    terminal_decision: str | None
    lead_chunks: int | None
    early_warning: bool
    warning_at_or_before_failure: bool
    missed_warning: bool
    decision_counts: dict[str, int]


def _policy_uncertainty_from_episode_step(step) -> float:
    stats = step.observation.policy_stats
    if len(stats) >= 4:
        return float(np.clip(stats[3], 0.0, 1.0))
    return 0.0


def _summarize_replay(strategy: str, rows: list[EpisodeReplayResult]) -> dict[str, Any]:
    if not rows:
        return {'strategy': strategy, 'episodes': 0}
    failure_rows = [r for r in rows if r.failure_onset is not None]
    leads = [r.lead_chunks for r in failure_rows if r.lead_chunks is not None]
    positive_leads = [v for v in leads if v > 0]
    total_steps = sum(r.n_steps for r in rows)
    total_calls = sum(r.call_count for r in rows)
    decision_counts: dict[str, int] = {}
    for row in rows:
        for key, value in row.decision_counts.items():
            decision_counts[key] = decision_counts.get(key, 0) + int(value)
    return _clean({
        'strategy': strategy,
        'episodes': len(rows),
        'failure_episodes': len(failure_rows),
        'total_steps': total_steps,
        'total_verifier_calls': total_calls,
        'mean_calls_per_episode': total_calls / max(len(rows), 1),
        'verifier_call_rate': total_calls / max(total_steps, 1),
        'warning_coverage': sum(r.warning_at_or_before_failure for r in failure_rows) / max(len(failure_rows), 1),
        'early_warning_rate': sum(r.early_warning for r in failure_rows) / max(len(failure_rows), 1),
        'missed_warning_rate': sum(r.missed_warning for r in failure_rows) / max(len(failure_rows), 1),
        'mean_lead_chunks': float(np.mean(leads)) if leads else float('nan'),
        'median_lead_chunks': float(np.median(leads)) if leads else float('nan'),
        'mean_positive_lead_chunks': float(np.mean(positive_leads)) if positive_leads else float('nan'),
        'median_positive_lead_chunks': float(np.median(positive_leads)) if positive_leads else float('nan'),
        'terminal_episode_rate': sum(r.first_terminal_step is not None for r in rows) / max(len(rows), 1),
        'decision_counts': decision_counts,
    })


def _result_to_dict(row: EpisodeReplayResult) -> dict[str, Any]:
    return _clean({
        'strategy': row.strategy,
        'episode_id': row.episode_id,
        'n_steps': row.n_steps,
        'outcome': row.outcome,
        'failure_onset': row.failure_onset if row.failure_onset is not None else '',
        'call_count': row.call_count,
        'verifier_call_rate': row.verifier_call_rate,
        'first_alarm_step': row.first_alarm_step if row.first_alarm_step is not None else '',
        'first_terminal_step': row.first_terminal_step if row.first_terminal_step is not None else '',
        'terminal_decision': row.terminal_decision or '',
        'lead_chunks': row.lead_chunks if row.lead_chunks is not None else '',
        'early_warning': row.early_warning,
        'warning_at_or_before_failure': row.warning_at_or_before_failure,
        'missed_warning': row.missed_warning,
        'decision_counts': json.dumps(row.decision_counts, sort_keys=True),
    })


def _finalize_episode_replay(
    *,
    strategy: str,
    episode_id: str,
    n_steps: int,
    outcome: str,
    failure_onset: int | None,
    call_count: int,
    first_alarm_step: int | None,
    first_terminal_step: int | None,
    terminal_decision: str | None,
    decision_counts: dict[str, int],
) -> EpisodeReplayResult:
    lead_chunks = None
    if failure_onset is not None and first_alarm_step is not None:
        lead_chunks = int(failure_onset - first_alarm_step)
    early_warning = bool(lead_chunks is not None and lead_chunks > 0)
    warning_at_or_before = bool(lead_chunks is not None and lead_chunks >= 0)
    missed_warning = bool(failure_onset is not None and not warning_at_or_before)
    return EpisodeReplayResult(
        strategy=strategy,
        episode_id=episode_id,
        n_steps=n_steps,
        outcome=outcome,
        failure_onset=failure_onset,
        call_count=call_count,
        verifier_call_rate=call_count / max(n_steps, 1),
        first_alarm_step=first_alarm_step,
        first_terminal_step=first_terminal_step,
        terminal_decision=terminal_decision,
        lead_chunks=lead_chunks,
        early_warning=early_warning,
        warning_at_or_before_failure=warning_at_or_before,
        missed_warning=missed_warning,
        decision_counts=decision_counts,
    )


def _run_event_driven_episode(
    bundle: VerifierBundle,
    episode,
    config: AppConfig,
    *,
    episode_id: str,
    alarm_decisions: set[str],
    stop_on_terminal: bool,
) -> EpisodeReplayResult:
    pipeline = EventDrivenVerifierPipeline(bundle=bundle, config=config)
    summary = pipeline.run_episode(episode, stop_on_termination=stop_on_terminal)
    events = [e for e in summary.get('events', []) if e.get('triggered')]
    decision_counts: dict[str, int] = {}
    first_alarm: int | None = None
    first_terminal: int | None = None
    terminal_decision: str | None = summary.get('terminal_decision') or None
    for event in events:
        decision = str(event.get('decision') or '')
        if decision:
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        t = int(event.get('t', 0))
        if first_alarm is None and decision in alarm_decisions:
            first_alarm = t
        if first_terminal is None and event.get('terminated'):
            first_terminal = t
    return _finalize_episode_replay(
        strategy='event_driven',
        episode_id=episode_id,
        n_steps=len(episode.steps),
        outcome=episode.outcome,
        failure_onset=episode.failure_onset,
        call_count=len(events),
        first_alarm_step=first_alarm,
        first_terminal_step=first_terminal,
        terminal_decision=terminal_decision,
        decision_counts=decision_counts,
    )


def _run_fixed_rate_episode(
    bundle: VerifierBundle,
    episode,
    config: AppConfig,
    *,
    episode_id: str,
    rate: int,
    alarm_decisions: set[str],
    stop_on_terminal: bool,
) -> EpisodeReplayResult:
    local_config = AppConfig()
    local_config.encoder = config.encoder
    local_config.memory = config.memory
    local_config.watcher = config.watcher
    local_config.decision = config.decision
    local_config.training.horizons = tuple(bundle.horizons)
    dataset = build_feature_dataset([episode], local_config)
    if dataset.X.size == 0:
        return _finalize_episode_replay(
            strategy=f'fixed_rate_{rate}',
            episode_id=episode_id,
            n_steps=0,
            outcome=episode.outcome,
            failure_onset=episode.failure_onset,
            call_count=0,
            first_alarm_step=None,
            first_terminal_step=None,
            terminal_decision=None,
            decision_counts={},
        )
    _, _, done_prob, risk = bundle.predict(dataset.X.astype(np.float32))
    decider = DecisionLayer(config.decision)
    call_count = 0
    first_alarm: int | None = None
    first_terminal: int | None = None
    terminal_decision: str | None = None
    decision_counts: dict[str, int] = {}
    for idx in range(0, len(episode.steps), rate):
        step = episode.steps[idx]
        decision, _ = decider.decide(risk[idx], _policy_uncertainty_from_episode_step(step), float(done_prob[idx]))
        call_count += 1
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        t = int(step.observation.timestamp)
        if first_alarm is None and decision in alarm_decisions:
            first_alarm = t
        if decision in set(config.decision.terminal_decisions):
            first_terminal = t
            terminal_decision = decision
            if stop_on_terminal:
                break
    return _finalize_episode_replay(
        strategy=f'fixed_rate_{rate}',
        episode_id=episode_id,
        n_steps=len(episode.steps),
        outcome=episode.outcome,
        failure_onset=episode.failure_onset,
        call_count=call_count,
        first_alarm_step=first_alarm,
        first_terminal_step=first_terminal,
        terminal_decision=terminal_decision,
        decision_counts=decision_counts,
    )


def evaluate_replay_protocol(
    bundle_path: str | Path,
    shard_dir: str | Path,
    *,
    fixed_rates: tuple[int, ...] = (1, 3, 5, 10, 15),
    alarm_decisions: set[str] | None = None,
    stop_on_terminal: bool = True,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    bundle_path = Path(bundle_path)
    shard_dir = Path(shard_dir)
    if not bundle_path.exists():
        raise FileNotFoundError(f'Missing bundle: {bundle_path}')
    bundle = VerifierBundle.load(bundle_path)
    config = AppConfig()
    config.training.horizons = tuple(bundle.horizons)
    alarm_decisions = alarm_decisions or {'watch', 'shield', 'abstain'}
    eval_shards = list_hdf5_shards(shard_dir, split='eval')
    if not eval_shards:
        raise FileNotFoundError(f'No eval shards found under {shard_dir / "eval"}')
    strategy_rows: dict[str, list[EpisodeReplayResult]] = {'event_driven': []}
    for rate in fixed_rates:
        strategy_rows[f'fixed_rate_{rate}'] = []
    total_episode_count = 0
    for shard_idx, shard_path in enumerate(eval_shards):
        episodes = load_robomimic_hdf5(shard_path, config=config.dataset)
        for ep_idx, episode in enumerate(episodes):
            if max_episodes is not None and total_episode_count >= max_episodes:
                break
            episode_id = f'{Path(shard_path).name}:{ep_idx}'
            strategy_rows['event_driven'].append(
                _run_event_driven_episode(
                    bundle,
                    episode,
                    config,
                    episode_id=episode_id,
                    alarm_decisions=alarm_decisions,
                    stop_on_terminal=stop_on_terminal,
                )
            )
            for rate in fixed_rates:
                strategy_rows[f'fixed_rate_{rate}'].append(
                    _run_fixed_rate_episode(
                        bundle,
                        episode,
                        config,
                        episode_id=episode_id,
                        rate=rate,
                        alarm_decisions=alarm_decisions,
                        stop_on_terminal=stop_on_terminal,
                    )
                )
            total_episode_count += 1
        if max_episodes is not None and total_episode_count >= max_episodes:
            break
    summaries = [_summarize_replay(name, rows) for name, rows in strategy_rows.items()]
    by_name = {s['strategy']: s for s in summaries}
    event_calls = float(by_name.get('event_driven', {}).get('total_verifier_calls', 0.0))
    comparisons: list[dict[str, Any]] = []
    for rate in fixed_rates:
        name = f'fixed_rate_{rate}'
        fixed_calls = float(by_name.get(name, {}).get('total_verifier_calls', 0.0))
        comparisons.append({
            'baseline': name,
            'event_driven_call_reduction': 1.0 - event_calls / fixed_calls if fixed_calls > 0 else float('nan'),
            'event_driven_calls': event_calls,
            'fixed_rate_calls': fixed_calls,
        })
    episode_rows = []
    for rows in strategy_rows.values():
        episode_rows.extend(_result_to_dict(r) for r in rows)
    return _clean({
        'bundle': str(bundle_path),
        'shard_dir': str(shard_dir),
        'split': 'eval',
        'horizons': list(bundle.horizons),
        'fixed_rates': list(fixed_rates),
        'alarm_decisions': sorted(alarm_decisions),
        'stop_on_terminal': bool(stop_on_terminal),
        'summaries': summaries,
        'comparisons': comparisons,
        'episode_rows': episode_rows,
    })


def _write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def cmd_ablation(args: argparse.Namespace) -> int:
    specs = [_parse_bundle_spec(raw) for raw in args.bundle]
    variants = _parse_csv_strings(args.variants)
    result = evaluate_ablation_suite(specs, args.shard_dir, variants=variants, ece_bins=args.ece_bins)
    _write_json(args.output_json, result)
    _write_csv(args.output_csv, result['rows'])
    print(json.dumps({
        'output_json': str(args.output_json),
        'output_csv': str(args.output_csv),
        'bundles': result['bundles'],
        'num_rows': len(result['rows']),
    }, indent=2, ensure_ascii=False))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    rates = _parse_csv_ints(args.fixed_rates)
    alarm_decisions = set(_parse_csv_strings(args.alarm_decisions))
    result = evaluate_replay_protocol(
        args.bundle,
        args.shard_dir,
        fixed_rates=rates,
        alarm_decisions=alarm_decisions,
        stop_on_terminal=not args.no_stop_on_terminal,
        max_episodes=args.max_episodes,
    )
    _write_json(args.output_json, result)
    rows: list[dict[str, Any]] = []
    for row in result['summaries']:
        flat = dict(row)
        flat['row_type'] = 'summary'
        if isinstance(flat.get('decision_counts'), dict):
            flat['decision_counts'] = json.dumps(flat['decision_counts'], sort_keys=True)
        rows.append(flat)
    for row in result['comparisons']:
        flat = dict(row)
        flat['row_type'] = 'comparison'
        rows.append(flat)
    _write_csv(args.output_csv, rows)
    episode_csv = Path(str(args.output_csv).replace('.csv', '_episodes.csv'))
    _write_csv(episode_csv, result['episode_rows'])
    print(json.dumps({
        'output_json': str(args.output_json),
        'output_csv': str(args.output_csv),
        'episode_csv': str(episode_csv),
        'summaries': result['summaries'],
        'comparisons': result['comparisons'],
    }, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Offline replay and lightweight ablation evaluators for EDSV-FH.')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('ablation', help='evaluate raw/calibrated/monotonic horizon-risk variants without retraining')
    p.add_argument('--shard-dir', type=Path, required=True)
    p.add_argument('--bundle', action='append', required=True, help='name=/path/to/bundle.pkl; may be passed multiple times')
    p.add_argument('--variants', default='calibrated_monotonic,calibrated,raw_monotonic,raw')
    p.add_argument('--ece-bins', type=int, default=10)
    p.add_argument('--output-json', type=Path, required=True)
    p.add_argument('--output-csv', type=Path, required=True)
    p.set_defaults(func=cmd_ablation)

    p = sub.add_parser('replay', help='run offline event-by-event replay and fixed-rate verifier comparisons')
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--shard-dir', type=Path, required=True)
    p.add_argument('--fixed-rates', default='1,3,5,10,15')
    p.add_argument('--alarm-decisions', default='watch,shield,abstain')
    p.add_argument('--no-stop-on-terminal', action='store_true')
    p.add_argument('--max-episodes', type=int, default=None)
    p.add_argument('--output-json', type=Path, required=True)
    p.add_argument('--output-csv', type=Path, required=True)
    p.set_defaults(func=cmd_replay)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == '__main__':
    raise SystemExit(main())
