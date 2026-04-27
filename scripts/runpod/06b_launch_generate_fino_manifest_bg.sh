#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ensure_dirs
cd "${PROJECT_DIR}"
pid=$(launch_bg_named "06_generate_fino_manifest" "${FINO_MANIFEST_LOG}" bash scripts/runpod/06_generate_fino_manifest.sh)
echo "PID=${pid}"
echo "LOG=${FINO_MANIFEST_LOG}"
echo "PID_FILE=${BG_STATE_ROOT}/06_generate_fino_manifest.pid"
