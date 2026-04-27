#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
mkdir -p "$(dirname "${DROID_FAILURE_PSEUDO_MANIFEST_PATH}")" "$(dirname "${DROID_FAILURE_PSEUDO_CKPT}")"
cd "${PROJECT_DIR}"

if [[ "${USE_HF}" == "1" || "${ENCODER}" != "fallback" ]]; then
  export EDSVFH_DISABLE_CUDNN="${EDSVFH_DISABLE_CUDNN:-1}"
  bash scripts/runpod/00_repair_hf_runtime.sh
fi

if [[ ! -f "${DROID_SUCCESS_BASELINE_PATH}" ]]; then
  echo "DROID success baseline missing: ${DROID_SUCCESS_BASELINE_PATH}. Run 07a_fit_droid_success_baseline.sh first." >&2
  exit 1
fi
if [[ ! -f "${DROID_FAILURE_MANIFEST_PATH}" ]]; then
  echo "DROID failure manifest missing: ${DROID_FAILURE_MANIFEST_PATH}. Run 06_generate_droid_failure_manifest.sh first." >&2
  exit 1
fi
CMD=(
  python -u -m edsvfh.cli label-droid-failure-pseudo-onset
  --manifest "${DROID_FAILURE_MANIFEST_PATH}"
  --baseline "${DROID_SUCCESS_BASELINE_PATH}"
  --output "${DROID_FAILURE_PSEUDO_MANIFEST_PATH}"
  --image-size "${IMAGE_SIZE}"
  --encoder "${ENCODER}"
  --checkpoint "${DROID_FAILURE_PSEUDO_CKPT}"
  --checkpoint-every 32
)
run_timed "07b_label_droid_failure_pseudo_onset" "${CMD[@]}"
