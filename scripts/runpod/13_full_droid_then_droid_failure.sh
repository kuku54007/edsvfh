#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

# Prepared DROID RLDS -> curated shard conversion -> DROID pretrain ->
# DROID RLDS not-successful manifest -> pseudo-onset -> failure-horizon fine-tune -> demo

cd "${PROJECT_DIR}"

bash scripts/runpod/00_preflight.sh
bash scripts/runpod/12_verify_project.sh

export DROID_SOURCE="${DROID_SOURCE:-${DROID_RAW_ROOT}/1.0.1}"
if [[ ! -f "${DROID_SOURCE}/features.json" ]]; then
  echo "Prepared DROID RLDS not found at ${DROID_SOURCE}." >&2
  echo "Download or point DROID_SOURCE to a TFDS builder directory containing features.json." >&2
  exit 2
fi

if should_force_rerun || ! dir_has_hdf5 "${DROID_CURATED_ROOT}"; then
  bash scripts/runpod/04_convert_droid_curated.sh
else
  log_step "SKIP DROID curated conversion; HDF5 shards already exist under ${DROID_CURATED_ROOT}"
fi

if should_force_rerun || [[ ! -f "${BASE_BUNDLE}" ]]; then
  bash scripts/runpod/05_train_droid_curated.sh
else
  log_step "SKIP DROID base training; found ${BASE_BUNDLE}"
fi

bash scripts/runpod/09_demo_droid_debug.sh || true

export DROID_FAILURE_SOURCE_MODE="${DROID_FAILURE_SOURCE_MODE:-rlds}"
export DROID_FAILURE_RLDS_ROOT="${DROID_FAILURE_RLDS_ROOT:-${DROID_SOURCE}}"
bash scripts/runpod/07c_run_droid_failure_pseudo_onset_pipeline.sh
bash scripts/runpod/10_demo_droid_failure.sh || true
