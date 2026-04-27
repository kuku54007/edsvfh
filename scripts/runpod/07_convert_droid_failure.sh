#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
mkdir -p "${DROID_FAILURE_CONVERTED_ROOT}" "$(dirname "${DROID_FAILURE_CONVERT_CKPT}")"
cd "${PROJECT_DIR}"

MANIFEST="${DROID_FAILURE_MANIFEST_PATH}"
PREFER_PSEUDO=0
if [[ "${USE_PSEUDO_ONSET}" != "0" && -f "${DROID_FAILURE_PSEUDO_MANIFEST_PATH}" ]]; then
  MANIFEST="${DROID_FAILURE_PSEUDO_MANIFEST_PATH}"
  PREFER_PSEUDO=1
fi
CMD=(
  python -u -m edsvfh.cli convert-droid-failure-manifest
  --manifest "${MANIFEST}"
  --output-dir "${DROID_FAILURE_CONVERTED_ROOT}"
  --episodes-per-shard 32
  --image-size "${IMAGE_SIZE}"
  --checkpoint "${DROID_FAILURE_CONVERT_CKPT}"
  --checkpoint-every 1
)
if [[ "${PREFER_PSEUDO}" == "1" ]]; then
  CMD+=(--prefer-pseudo-onset)
fi
run_timed "07_convert_droid_failure" "${CMD[@]}"
