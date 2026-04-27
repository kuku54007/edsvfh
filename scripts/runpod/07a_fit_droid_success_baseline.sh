#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
mkdir -p "$(dirname "${DROID_SUCCESS_BASELINE_PATH}")"
if ! BASELINE_SHARD_DIR="$(resolve_droid_baseline_shard_dir)"; then
  echo "No DROID shards found under ${DROID_CURATED_ROOT} or ${DROID_DEBUG_ROOT}; cannot fit success baseline." >&2
  exit 1
fi
cd "${PROJECT_DIR}"
if [[ "${USE_HF}" == "1" || "${ENCODER}" != "fallback" ]]; then
  export EDSVFH_DISABLE_CUDNN="${EDSVFH_DISABLE_CUDNN:-1}"
  bash scripts/runpod/00_repair_hf_runtime.sh
fi
CMD=(
  python -u -m edsvfh.cli fit-droid-success-baseline
  --shard-dir "${BASELINE_SHARD_DIR}"
  --output "${DROID_SUCCESS_BASELINE_PATH}"
  --encoder "${ENCODER}"
  --feature-source "${PSEUDO_ONSET_FEATURE_SOURCE}"
  --window "${PSEUDO_ONSET_WINDOW}"
  --phase-bins "${PSEUDO_ONSET_PHASE_BINS}"
  --quantile "${PSEUDO_ONSET_QUANTILE}"
  --min-phase-count "${PSEUDO_ONSET_MIN_PHASE_COUNT}"
)
if [[ -n "${PSEUDO_ONSET_FIT_MAX_EPISODES}" ]]; then
  CMD+=(--max-episodes "${PSEUDO_ONSET_FIT_MAX_EPISODES}")
fi
run_timed "07a_fit_droid_success_baseline" "${CMD[@]}"
