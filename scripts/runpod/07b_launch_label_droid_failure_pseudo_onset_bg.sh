#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ensure_dirs
cd "${PROJECT_DIR}"
pid=$(launch_bg_named "07b_label_droid_failure_pseudo_onset" "${DROID_FAILURE_PSEUDO_LABEL_LOG}" bash scripts/runpod/07b_label_droid_failure_pseudo_onset.sh)
echo "PID=${pid}"
echo "LOG=${DROID_FAILURE_PSEUDO_LABEL_LOG}"
echo "PID_FILE=${BG_STATE_ROOT}/07b_label_droid_failure_pseudo_onset.pid"
