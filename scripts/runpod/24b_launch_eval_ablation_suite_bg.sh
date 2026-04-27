#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
cd "${PROJECT_DIR}"
pid="$(launch_bg_named ablation_suite_eval "${ABLATION_LOG}" bash scripts/runpod/24_eval_ablation_suite.sh)"
cat <<MSG
Started lightweight ablation-suite evaluation in background.
PID=${pid}
PID_FILE=${BG_STATE_ROOT}/ablation_suite_eval.pid
LOG=${ABLATION_LOG}
OUTPUT_JSON=${ABLATION_OUTPUT_JSON}
OUTPUT_CSV=${ABLATION_OUTPUT_CSV}
BUNDLES=${ABLATION_BUNDLES}
VARIANTS=${ABLATION_VARIANTS}

Watch it with:
  bash scripts/runpod/17_status_and_logs.sh
  tail -f ${ABLATION_LOG}
MSG
