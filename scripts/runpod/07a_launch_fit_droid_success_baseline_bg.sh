#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ensure_dirs
cd "${PROJECT_DIR}"
pid=$(launch_bg_named "07a_fit_droid_success_baseline" "${DROID_SUCCESS_BASELINE_LOG}" bash scripts/runpod/07a_fit_droid_success_baseline.sh)
echo "PID=${pid}"
echo "LOG=${DROID_SUCCESS_BASELINE_LOG}"
echo "PID_FILE=${BG_STATE_ROOT}/07a_fit_droid_success_baseline.pid"
