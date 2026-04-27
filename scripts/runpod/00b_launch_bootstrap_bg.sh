#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ensure_dirs
cd "${PROJECT_DIR}"
pid="$(launch_bg_named bootstrap "${LOG_ROOT}/bootstrap.log" bash scripts/runpod/00_bootstrap.sh)"
cat <<MSG
Started bootstrap in background.
PID=${pid}
PID_FILE=${BG_STATE_ROOT}/bootstrap.pid
LOG=${LOG_ROOT}/bootstrap.log
MSG
