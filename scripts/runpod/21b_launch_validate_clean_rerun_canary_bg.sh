#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
pid=$(launch_bg_named "21_validate_clean_rerun_canary" "${CANARY_LOG}" bash scripts/runpod/21_validate_clean_rerun_canary.sh)
echo "PID=${pid}"
echo "PID_FILE=${BG_STATE_ROOT}/21_validate_clean_rerun_canary.pid"
echo "LOG=${CANARY_LOG}"
echo "SUMMARY_JSON=${CANARY_SUMMARY_JSON}"
echo "SUMMARY_TXT=${CANARY_SUMMARY_TXT}"
