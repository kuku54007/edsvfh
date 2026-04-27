#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
cd "${PROJECT_DIR}"
export EDSVFH_TF_DISABLE_GPU="${EDSVFH_TF_DISABLE_GPU:-1}"
CONVERT_ENCODER="${CONVERT_ENCODER:-${ENCODER}}"
CONVERT_DEVICE="${CONVERT_DEVICE:-${EDSVFH_DEVICE:-cuda}}"
if [[ "${USE_HF}" == "1" || "${CONVERT_ENCODER}" != "fallback" ]]; then
  export EDSVFH_DISABLE_CUDNN="${EDSVFH_DISABLE_CUDNN:-0}"
  export EDSVFH_CONVERT_BATCH_SIZE="${CONVERT_BATCH_SIZE}"
  bash scripts/runpod/00_repair_hf_runtime.sh
fi
run_timed "02_convert_droid100" python -u -m edsvfh.cli convert-droid \
  --source "${DROID_RAW_ROOT}/droid_100" \
  --output-dir "${DROID_DEBUG_ROOT}" \
  --episodes-per-shard 32 \
  --image-size "${IMAGE_SIZE}" \
  --step-stride 2 \
  --action-space raw_action \
  --precompute-encoder "${CONVERT_ENCODER}" \
  --precompute-device "${CONVERT_DEVICE}" \
  --checkpoint-every 32
