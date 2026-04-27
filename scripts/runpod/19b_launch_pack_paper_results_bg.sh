#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ensure_dirs
cd "${PROJECT_DIR}"
pid=$(launch_bg_named "19_pack_paper_results" "${PAPER_PACK_LOG}" bash scripts/runpod/19_pack_paper_results.sh)
echo "PID=${pid}"
echo "LOG=${PAPER_PACK_LOG}"
echo "PID_FILE=${BG_STATE_ROOT}/19_pack_paper_results.pid"
