#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ensure_dirs
cd "${PROJECT_DIR}"
pid="$(launch_bg_named smoke_test_full_paper_path_droid_failure "${LOG_ROOT}/smoke_test_full_paper_path_droid_failure.log" bash scripts/runpod/30_smoke_test_full_paper_path_droid_failure.sh)"
cat <<MSG
Started full smoke test in background.
PID=${pid}
PID_FILE=${BG_STATE_ROOT}/smoke_test_full_paper_path_droid_failure.pid
LOG=${LOG_ROOT}/smoke_test_full_paper_path_droid_failure.log
MSG
