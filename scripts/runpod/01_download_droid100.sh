#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
if ! command -v gsutil >/dev/null 2>&1; then
  run_timed "01_download_droid100-pip" python -u -m pip install gsutil
fi
if [[ -d "${DROID_RAW_ROOT}/droid_100" ]]; then
  echo "DROID_100 already exists at ${DROID_RAW_ROOT}/droid_100"
  exit 0
fi

gsutil -m cp -r gs://gresearch/robotics/droid_100 "${DROID_RAW_ROOT}"
