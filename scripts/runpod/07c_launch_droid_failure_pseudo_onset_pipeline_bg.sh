#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ensure_dirs
cd "${PROJECT_DIR}"
pid=$(launch_bg_named "07c_droid_failure_pseudo_onset_pipeline" "${DROID_FAILURE_PSEUDO_PIPELINE_LOG}" bash scripts/runpod/07c_run_droid_failure_pseudo_onset_pipeline.sh)
echo "PID=${pid}"
echo "LOG=${DROID_FAILURE_PSEUDO_PIPELINE_LOG}"
echo "PID_FILE=${BG_STATE_ROOT}/07c_droid_failure_pseudo_onset_pipeline.pid"
