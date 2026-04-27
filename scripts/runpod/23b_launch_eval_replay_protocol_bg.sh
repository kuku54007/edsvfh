#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
cd "${PROJECT_DIR}"
pid="$(launch_bg_named replay_protocol_eval "${REPLAY_LOG}" bash scripts/runpod/23_eval_replay_protocol.sh)"
cat <<MSG
Started offline replay protocol evaluation in background.
PID=${pid}
PID_FILE=${BG_STATE_ROOT}/replay_protocol_eval.pid
LOG=${REPLAY_LOG}
OUTPUT_JSON=${REPLAY_OUTPUT_JSON}
OUTPUT_CSV=${REPLAY_OUTPUT_CSV}
EPISODE_CSV=${REPLAY_OUTPUT_CSV%.csv}_episodes.csv
BUNDLE=${REPLAY_BUNDLE}
FIXED_RATES=${REPLAY_FIXED_RATES}

Watch it with:
  bash scripts/runpod/17_status_and_logs.sh
  tail -f ${REPLAY_LOG}
MSG
