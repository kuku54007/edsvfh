#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
mkdir -p "$(dirname "${FINO_PSEUDO_MANIFEST_PATH}")" "$(dirname "${FINO_PSEUDO_CKPT}")"
cd "${PROJECT_DIR}"
if [[ ! -f "${FINO_MANIFEST_PATH}" ]]; then
  echo "FINO manifest not found: ${FINO_MANIFEST_PATH}" >&2
  exit 1
fi
if [[ ! -f "${DROID_SUCCESS_BASELINE_PATH}" ]]; then
  echo "DROID success baseline not found: ${DROID_SUCCESS_BASELINE_PATH}" >&2
  exit 1
fi
if [[ "${USE_HF}" == "1" || "${ENCODER}" != "fallback" ]]; then
  export EDSVFH_DISABLE_CUDNN="${EDSVFH_DISABLE_CUDNN:-1}"
  bash scripts/runpod/00_repair_hf_runtime.sh
fi
CMD=(
  python -u -m edsvfh.cli label-fino-pseudo-onset
  --manifest "${FINO_MANIFEST_PATH}"
  --baseline "${DROID_SUCCESS_BASELINE_PATH}"
  --output "${FINO_PSEUDO_MANIFEST_PATH}"
  --image-size "${IMAGE_SIZE}"
  --encoder "${ENCODER}"
  --checkpoint "${FINO_PSEUDO_CKPT}"
  --checkpoint-every 32
)
run_timed "07b_label_fino_pseudo_onset" "${CMD[@]}"
