#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ensure_dirs
cd "${PROJECT_DIR}"
pid=$(launch_bg_named "06_generate_droid_failure_manifest" "${DROID_FAILURE_MANIFEST_LOG}" bash scripts/runpod/06_generate_droid_failure_manifest.sh)
echo "PID=${pid}"
echo "LOG=${DROID_FAILURE_MANIFEST_LOG}"
echo "PID_FILE=${BG_STATE_ROOT}/06_generate_droid_failure_manifest.pid"
