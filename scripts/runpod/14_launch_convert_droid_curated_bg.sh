#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ensure_dirs
cd "${PROJECT_DIR}"
pid=$(launch_bg_named "convert_droid_curated" "${CONVERT_DROID_CURATED_LOG}" bash scripts/runpod/04_convert_droid_curated.sh)
echo "PID=${pid}"
echo "LOG=${CONVERT_DROID_CURATED_LOG}"
echo "PID_FILE=${BG_STATE_ROOT}/convert_droid_curated.pid"
