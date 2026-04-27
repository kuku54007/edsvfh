#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
cd "${PROJECT_DIR}"
if [[ "${USE_HF}" == "1" || "${ENCODER}" != "fallback" ]]; then
  export EDSVFH_DISABLE_CUDNN="${EDSVFH_DISABLE_CUDNN:-1}"
  bash scripts/runpod/00_repair_hf_runtime.sh
fi
run_timed "03_train_droid_debug" python -u -m edsvfh.cli train-sharded \
  --shard-dir "${DROID_DEBUG_ROOT}" \
  --output "${ARTIFACT_ROOT}/droid_debug_bundle.pkl" \
  --encoder "${ENCODER}" \
  --checkpoint "${DROID_DEBUG_CKPT}" \
  --checkpoint-every 1
