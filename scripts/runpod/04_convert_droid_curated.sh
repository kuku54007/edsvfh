#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
DROID_SOURCE="${DROID_SOURCE:-${DROID_RAW_ROOT}/1.0.1}"
cd "${PROJECT_DIR}"
export EDSVFH_TF_DISABLE_GPU="${EDSVFH_TF_DISABLE_GPU:-1}"
CONVERT_ENCODER="${CONVERT_ENCODER:-${ENCODER}}"
CONVERT_DEVICE="${CONVERT_DEVICE:-${EDSVFH_DEVICE:-cuda}}"
if [[ "${USE_HF}" == "1" || "${CONVERT_ENCODER}" != "fallback" ]]; then
  export EDSVFH_DISABLE_CUDNN="${EDSVFH_DISABLE_CUDNN:-0}"
  export EDSVFH_CONVERT_BATCH_SIZE="${CONVERT_BATCH_SIZE}"
  bash scripts/runpod/00_repair_hf_runtime.sh
fi
CMD=(
  python -u -m edsvfh.cli convert-droid
  --source "${DROID_SOURCE}"
  --output-dir "${DROID_CURATED_ROOT}"
  --episodes-per-shard "${EPISODES_PER_SHARD}"
  --image-size "${IMAGE_SIZE}"
  --step-stride "${STEP_STRIDE}"
  --action-space "${ACTION_SPACE}"
  --outcome-filter "${DROID_OUTCOME_FILTER:-success}"
  --precompute-encoder "${CONVERT_ENCODER}"
  --precompute-device "${CONVERT_DEVICE}"
  --checkpoint-every 64
)
if [[ -n "${MAX_EPISODES:-}" ]]; then
  CMD+=(--max-episodes "${MAX_EPISODES}")
fi
run_timed "04_convert_droid_curated" "${CMD[@]}"
