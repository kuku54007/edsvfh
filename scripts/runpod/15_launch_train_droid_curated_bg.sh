#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ensure_dirs
cd "${PROJECT_DIR}"
pid=$(launch_bg_named "train_droid_curated" "${TRAIN_DROID_CURATED_LOG}" bash scripts/runpod/05_train_droid_curated.sh)
echo "PID=${pid}"
echo "LOG=${TRAIN_DROID_CURATED_LOG}"
echo "PID_FILE=${BG_STATE_ROOT}/train_droid_curated.pid"
