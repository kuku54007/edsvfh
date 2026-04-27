#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ensure_dirs
cd "${PROJECT_DIR}"
pid=$(launch_bg_named "07_convert_droid_failure" "${DROID_FAILURE_CONVERT_LOG}" bash scripts/runpod/07_convert_droid_failure.sh)
echo "PID=${pid}"
echo "LOG=${DROID_FAILURE_CONVERT_LOG}"
echo "PID_FILE=${BG_STATE_ROOT}/07_convert_droid_failure.pid"
