#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
mkdir -p "$(dirname "${FINO_MANIFEST_PATH}")" "$(dirname "${FINO_MANIFEST_CKPT}")"
cd "${PROJECT_DIR}"
run_timed "06_generate_fino_manifest" python -u -m edsvfh.cli generate-fino-manifest   --root-dir "${FINO_RAW_ROOT}"   --output "${FINO_MANIFEST_PATH}"   --checkpoint "${FINO_MANIFEST_CKPT}"   --checkpoint-every 32
