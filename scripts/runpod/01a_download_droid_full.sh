#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ensure_dirs
if ! command -v gsutil >/dev/null 2>&1; then
  run_timed "01a_download_droid_full-pip" python -u -m pip install gsutil
fi
mkdir -p "${DROID_RAW_ROOT}"
gsutil -m cp -r gs://gresearch/robotics/droid/1.0.1 "${DROID_RAW_ROOT}"
