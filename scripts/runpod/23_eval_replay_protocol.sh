#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
cd "${PROJECT_DIR}"

log_step "START 23_eval_replay_protocol"
print_layout

export EDSVFH_HF_LOCAL_ONLY="${EDSVFH_HF_LOCAL_ONLY:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ ! -f "${REPLAY_BUNDLE}" ]]; then
  echo "Missing replay bundle: ${REPLAY_BUNDLE}" >&2
  echo "Run the main or extended-horizon DROID failure experiment first." >&2
  exit 1
fi
if ! dir_has_hdf5 "${DROID_FAILURE_CONVERTED_ROOT}/eval"; then
  echo "Missing DROID failure eval shards under ${DROID_FAILURE_CONVERTED_ROOT}/eval" >&2
  exit 1
fi

CMD=(
  python -u -m edsvfh.eval_protocols replay
  --bundle "${REPLAY_BUNDLE}"
  --shard-dir "${DROID_FAILURE_CONVERTED_ROOT}"
  --fixed-rates "${REPLAY_FIXED_RATES}"
  --alarm-decisions "${REPLAY_ALARM_DECISIONS}"
  --output-json "${REPLAY_OUTPUT_JSON}"
  --output-csv "${REPLAY_OUTPUT_CSV}"
)
if [[ "${REPLAY_STOP_ON_TERMINAL}" == "0" ]]; then
  CMD+=(--no-stop-on-terminal)
fi
if [[ -n "${REPLAY_MAX_EPISODES}" ]]; then
  CMD+=(--max-episodes "${REPLAY_MAX_EPISODES}")
fi

run_timed "23_eval_replay_protocol" "${CMD[@]}"
log_step "END 23_eval_replay_protocol"
