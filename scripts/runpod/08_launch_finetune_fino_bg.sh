#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ensure_dirs
cd "${PROJECT_DIR}"
pid=$(launch_bg_named "08_finetune_fino" "${FINO_TRAIN_LOG}" bash scripts/runpod/08_finetune_fino.sh)
echo "PID=${pid}"
echo "LOG=${FINO_TRAIN_LOG}"
echo "PID_FILE=${BG_STATE_ROOT}/08_finetune_fino.pid"
