#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
pid=$(launch_bg_named "20_clean_rerun_fino_pseudo_onset" "${LOG_ROOT}/clean_rerun_fino_pseudo_onset.log" bash scripts/runpod/20_clean_rerun_fino_pseudo_onset.sh)
echo "PID=${pid}"
echo "PID_FILE=${BG_STATE_ROOT}/20_clean_rerun_fino_pseudo_onset.pid"
echo "LOG=${LOG_ROOT}/clean_rerun_fino_pseudo_onset.log"
