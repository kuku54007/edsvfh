#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
mkdir -p "$(dirname "${FINO_CKPT}")" "$(dirname "${ARTIFACT_ROOT}/droid_fino_bundle.pkl")"
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
  python -u -m edsvfh.cli fine-tune-fino
  --base-bundle "${BASE_BUNDLE_PATH}"
  --shard-dir "${FINO_CONVERTED_ROOT}"
  --output "${ARTIFACT_ROOT}/droid_fino_bundle.pkl"
  --encoder "${ENCODER}"
  --epochs "${EPOCHS}"
  --horizons "${HORIZONS}"
  --checkpoint "${FINO_CKPT}"
  --checkpoint-every 1
)
if [[ "${FINO_UPDATE_SCALER}" != "0" ]]; then
  CMD+=(--update-scaler)
fi
run_timed "08_finetune_fino" "${CMD[@]}"
