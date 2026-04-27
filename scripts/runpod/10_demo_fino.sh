#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
cd "${PROJECT_DIR}"
run_timed "10_demo_fino" python -u -m edsvfh.cli demo \
  --bundle "${ARTIFACT_ROOT}/droid_fino_bundle.pkl" \
  --dataset "${FINO_CONVERTED_ROOT}/eval/fino_eval_0000.hdf5" \
  --episode-index "${EPISODE_INDEX:-0}"
