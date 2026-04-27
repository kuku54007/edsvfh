#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
cd "${PROJECT_DIR}"
run_timed "09_demo_droid_debug" python -u -m edsvfh.cli demo \
  --bundle "${ARTIFACT_ROOT}/droid_debug_bundle.pkl" \
  --dataset "${DROID_DEBUG_ROOT}/eval/droid_eval_0000.hdf5" \
  --episode-index "${EPISODE_INDEX:-0}"
