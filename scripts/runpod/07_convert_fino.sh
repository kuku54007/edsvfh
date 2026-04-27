#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
mkdir -p "${FINO_CONVERTED_ROOT}" "$(dirname "${FINO_CONVERT_CKPT}")"
cd "${PROJECT_DIR}"
MANIFEST_TO_USE="${FINO_MANIFEST_PATH}"
EXTRA_ARGS=()
if [[ "${USE_PSEUDO_ONSET}" != "0" ]]; then
  if [[ -f "${FINO_PSEUDO_MANIFEST_PATH}" ]]; then
    MANIFEST_TO_USE="${FINO_PSEUDO_MANIFEST_PATH}"
    EXTRA_ARGS+=(--prefer-pseudo-onset)
  else
    echo "Pseudo-onset manifest not found at ${FINO_PSEUDO_MANIFEST_PATH}; falling back to ${FINO_MANIFEST_PATH}." >&2
  fi
fi
CMD=(
  python -u -m edsvfh.cli convert-fino-manifest
  --manifest "${MANIFEST_TO_USE}"
  --output-dir "${FINO_CONVERTED_ROOT}"
  --episodes-per-shard 32
  --image-size "${IMAGE_SIZE}"
  --checkpoint "${FINO_CONVERT_CKPT}"
  --checkpoint-every 16
)
CMD+=("${EXTRA_ARGS[@]}")
run_timed "07_convert_fino" "${CMD[@]}"
