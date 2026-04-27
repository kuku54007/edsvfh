#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ensure_dirs
cd "${PROJECT_DIR}"
pid="$(launch_bg_named full_paper_path_droid_failure "${LOG_ROOT}/full_paper_path_droid_failure.log" bash scripts/runpod/25_full_paper_path_droid_failure.sh)"
cat <<MSG
Started full paper path in background.
PID=${pid}
PID_FILE=${BG_STATE_ROOT}/full_paper_path_droid_failure.pid
LOG=${LOG_ROOT}/full_paper_path_droid_failure.log
MSG
