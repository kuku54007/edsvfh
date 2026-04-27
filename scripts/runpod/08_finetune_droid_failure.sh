#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
mkdir -p "$(dirname "${DROID_FAILURE_CKPT}")" "$(dirname "${DROID_FAILURE_BUNDLE}")"
if ! BASE_BUNDLE_PATH="$(resolve_base_bundle_path)"; then
  echo "No base DROID bundle found. Checked: ${BASE_BUNDLE}, ${ARTIFACT_ROOT}/droid_curated_bundle.pkl, ${ARTIFACT_ROOT}/droid_debug_bundle.pkl" >&2
  exit 1
fi
cd "${PROJECT_DIR}"
if [[ "${USE_HF}" == "1" || "${ENCODER}" != "fallback" ]]; then
  export EDSVFH_DISABLE_CUDNN="${EDSVFH_DISABLE_CUDNN:-1}"
  bash scripts/runpod/00_repair_hf_runtime.sh
fi
CMD=(
  python -u -m edsvfh.cli fine-tune-droid-failure
  --base-bundle "${BASE_BUNDLE_PATH}"
  --shard-dir "${DROID_FAILURE_CONVERTED_ROOT}"
  --output "${DROID_FAILURE_BUNDLE}"
  --encoder "${ENCODER}"
  --epochs "${EPOCHS}"
  --horizons "${HORIZONS}"
  --checkpoint "${DROID_FAILURE_CKPT}"
  --checkpoint-every 1
)
if [[ "${DROID_FAILURE_UPDATE_SCALER}" != "0" ]]; then
  CMD+=(--update-scaler)
fi
run_timed "08_finetune_droid_failure" "${CMD[@]}"
