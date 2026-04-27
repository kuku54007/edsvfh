#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
cd "${PROJECT_DIR}"
pid="$(launch_bg_named larger_horizons_test "${LARGER_HORIZONS_LOG}" bash scripts/runpod/22_test_larger_horizons.sh)"
cat <<MSG
Started larger-horizons test in background.
PID=${pid}
PID_FILE=${BG_STATE_ROOT}/larger_horizons_test.pid
LOG=${LARGER_HORIZONS_LOG}
OUTPUT_BUNDLE=${LARGER_HORIZONS_OUTPUT_BUNDLE}
BASE_BUNDLE=${LARGER_HORIZONS_BASE_BUNDLE}
FREEZE_EXISTING=${LARGER_HORIZONS_FREEZE_EXISTING}
HORIZONS=${LARGER_HORIZONS}

Watch it with:
  bash scripts/runpod/17_status_and_logs.sh
  tail -f ${LARGER_HORIZONS_LOG}
MSG
