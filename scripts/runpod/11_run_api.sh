#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
cd "${PROJECT_DIR}"
run_timed "11_run_api" python -u -m edsvfh.cli serve \
  --bundle "${BUNDLE_PATH:-${ARTIFACT_ROOT}/droid_fino_bundle.pkl}" \
  --host 0.0.0.0 \
  --port "${PORT:-8000}"
