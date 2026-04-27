#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

# This script follows the paper-oriented path:
# full DROID RLDS -> curated shard conversion -> DROID pretrain -> FINO manifest/convert -> FINO fine-tune -> online demos

cd "${PROJECT_DIR}"

bash scripts/runpod/00_preflight.sh
bash scripts/runpod/12_verify_project.sh
bash scripts/runpod/01a_download_droid_full.sh

# Smoke test the full-source path first to avoid wasting GPU/storage budget.
export DROID_SOURCE="${DROID_SOURCE:-${DROID_RAW_ROOT}/1.0.1}"
export MAX_EPISODES="${MAX_EPISODES:-32}"
bash scripts/runpod/04_convert_droid_curated.sh
bash scripts/runpod/05_train_droid_curated.sh
bash scripts/runpod/09_demo_droid_debug.sh || true

# If FINO raw is present, continue to failure-side fine-tuning.
if [[ -d "${FINO_RAW_ROOT}" && -n "$(find "${FINO_RAW_ROOT}" -mindepth 1 -maxdepth 1 2>/dev/null)" ]]; then
  bash scripts/runpod/07c_run_fino_pseudo_onset_pipeline.sh
  bash scripts/runpod/10_demo_fino.sh
else
  echo "FINO raw data not found under ${FINO_RAW_ROOT}; skipping FINO steps."
fi
