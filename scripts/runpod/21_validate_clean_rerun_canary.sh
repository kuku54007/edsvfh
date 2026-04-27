#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
mkdir -p "$(dirname "${CANARY_LOG}")" "$(dirname "${CANARY_SUMMARY_JSON}")" "$(dirname "${CANARY_SUMMARY_TXT}")"
cd "${PROJECT_DIR}"

log_step "START clean rerun canary"
print_layout
log_step "CANARY_PSEUDO_ONSET_FIT_MAX_EPISODES=${CANARY_PSEUDO_ONSET_FIT_MAX_EPISODES}"
log_step "CANARY_EPOCHS=${CANARY_EPOCHS}"
log_step "CANARY_RUN_TARGETED_TESTS=${CANARY_RUN_TARGETED_TESTS}"
log_step "CANARY_RUN_PIPELINE=${CANARY_RUN_PIPELINE}"
log_step "CANARY_STRICT=${CANARY_STRICT}"

if [[ "${CANARY_RUN_TARGETED_TESTS}" != "0" ]]; then
  run_timed "21_canary_targeted_tests" \
    python -m pytest -q \
      tests/test_fiper_pseudo_onset.py::test_convert_fino_manifest_rebuilds_when_manifest_changes \
      tests/test_fiper_pseudo_onset.py::test_pseudo_onset_keeps_original_when_low_confidence \
      tests/test_fiper_pseudo_onset.py::test_rebuild_fino_pseudo_onset_runs_end_to_end
fi

export CANARY_SUMMARY_JSON CANARY_SUMMARY_TXT CANARY_STRICT CANARY_RUN_PIPELINE CANARY_RUN_TARGETED_TESTS FINO_PSEUDO_MANIFEST_PATH FINO_CONVERT_CKPT ARTIFACT_ROOT

if [[ "${CANARY_RUN_PIPELINE}" != "0" ]]; then
  export USE_PSEUDO_ONSET=1
  export PSEUDO_ONSET_FIT_MAX_EPISODES="${CANARY_PSEUDO_ONSET_FIT_MAX_EPISODES}"
  export EPOCHS="${CANARY_EPOCHS}"
  export FINO_UPDATE_SCALER="${FINO_UPDATE_SCALER}"
  run_timed "21_canary_clean_rerun_pipeline" bash scripts/runpod/20_clean_rerun_fino_pseudo_onset.sh
fi

python - <<'PY'
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

summary_path = Path(os.environ['CANARY_SUMMARY_JSON'])
summary_txt = Path(os.environ['CANARY_SUMMARY_TXT'])
strict = os.environ.get('CANARY_STRICT', '1') != '0'
run_pipeline = os.environ.get('CANARY_RUN_PIPELINE', '1') != '0'
run_tests = os.environ.get('CANARY_RUN_TARGETED_TESTS', '1') != '0'

pseudo_manifest = Path(os.environ['FINO_PSEUDO_MANIFEST_PATH'])
convert_ckpt = Path(os.environ['FINO_CONVERT_CKPT'])
output_bundle = Path(os.environ['ARTIFACT_ROOT']) / 'droid_fino_bundle.pkl'
expected_source = str(pseudo_manifest)

summary: dict[str, object] = {
    'targeted_tests_requested': run_tests,
    'pipeline_requested': run_pipeline,
    'strict_mode': strict,
    'expected_pseudo_manifest': expected_source,
    'checks': {},
}
checks: dict[str, object] = summary['checks']  # type: ignore[assignment]
errors: list[str] = []

if run_pipeline:
    checks['pseudo_manifest_exists'] = pseudo_manifest.exists()
    if not pseudo_manifest.exists():
        errors.append(f'missing pseudo manifest: {pseudo_manifest}')
    else:
        rows = [json.loads(line) for line in pseudo_manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
        failures = [row for row in rows if str(row.get('outcome', '')).lower() != 'success']
        zero_count = sum(1 for row in failures if int(row.get('pseudo_failure_onset', -1)) == 0)
        reason_counts = Counter(str(row.get('pseudo_onset_reason', '<missing>')) for row in failures)
        all_conf = [float(row.get('pseudo_onset_confidence', 0.0)) for row in failures if row.get('pseudo_onset_confidence') is not None]
        checks['failure_episodes'] = len(failures)
        checks['zero_pseudo_onset'] = zero_count
        checks['reason_counts'] = dict(reason_counts)
        checks['all_zero_onset'] = bool(failures) and zero_count == len(failures)
        checks['all_failure_peak_fallback'] = bool(reason_counts) and set(reason_counts) == {'failure_peak_fallback'}
        if all_conf:
            checks['confidence_min'] = min(all_conf)
            checks['confidence_max'] = max(all_conf)
        if failures and zero_count == len(failures):
            errors.append('all failure episodes collapsed to pseudo_failure_onset=0')
        if reason_counts and set(reason_counts) == {'failure_peak_fallback'}:
            errors.append('all pseudo-onset reasons are failure_peak_fallback')

    checks['convert_ckpt_exists'] = convert_ckpt.exists()
    if not convert_ckpt.exists():
        errors.append(f'missing convert checkpoint: {convert_ckpt}')
    else:
        obj = json.loads(convert_ckpt.read_text(encoding='utf-8'))
        checks['convert_source'] = obj.get('source')
        checks['convert_manifest_sha256_present'] = bool(obj.get('manifest_sha256'))
        checks['convert_resume_signature_present'] = bool(obj.get('resume_signature'))
        checks['convert_prefer_pseudo_onset'] = obj.get('prefer_pseudo_onset')
        if obj.get('source') != expected_source:
            errors.append(f'convert checkpoint source mismatch: {obj.get("source")} != {expected_source}')
        if not obj.get('manifest_sha256'):
            errors.append('convert checkpoint missing manifest_sha256')
        if not obj.get('resume_signature'):
            errors.append('convert checkpoint missing resume_signature')
        if obj.get('prefer_pseudo_onset') is not True:
            errors.append('convert checkpoint prefer_pseudo_onset is not true')

    checks['output_bundle_exists'] = output_bundle.exists()
    if not output_bundle.exists():
        errors.append(f'missing output bundle: {output_bundle}')

summary['errors'] = errors
summary['pass'] = not errors
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
summary_txt.write_text(
    '\n'.join([
        f'pass={summary["pass"]}',
        f'strict_mode={strict}',
        f'pipeline_requested={run_pipeline}',
        f'expected_pseudo_manifest={expected_source}',
        f'zero_pseudo_onset={checks.get("zero_pseudo_onset")}',
        f'reason_counts={checks.get("reason_counts")}',
        f'convert_source={checks.get("convert_source")}',
        f'convert_manifest_sha256_present={checks.get("convert_manifest_sha256_present")}',
        f'convert_resume_signature_present={checks.get("convert_resume_signature_present")}',
        f'output_bundle_exists={checks.get("output_bundle_exists")}',
        'errors=' + ('; '.join(errors) if errors else '<none>'),
    ]) + '\n',
    encoding='utf-8',
)
print(summary_txt.read_text(encoding='utf-8'), end='')
if strict and errors:
    raise SystemExit(2)
PY

log_step "END clean rerun canary"
