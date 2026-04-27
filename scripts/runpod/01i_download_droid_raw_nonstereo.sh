#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ensure_dirs
if ! command -v gsutil >/dev/null 2>&1; then
  run_timed "01i_download_droid_raw_nonstereo-pip" python -u -m pip install gsutil
fi
# Official docs document the raw bucket at gs://gresearch/robotics/droid_raw.
# Many users mirror a versioned subtree such as gs://gresearch/robotics/droid_raw/1.0.1.
DROID_RAW_GCS_PREFIX="${DROID_RAW_GCS_PREFIX:-gs://gresearch/robotics/droid_raw/1.0.1}"
TARGET_DIR="${DROID_FAILURE_RAW_ROOT}"
mkdir -p "${TARGET_DIR}"
# The v31 DROID-failure route only needs trajectory.h5 + metadata + non-stereo MP4s.
# Excluding SVO and stereo MP4 substantially reduces the transfer size.
gsutil -m rsync -r -x ".*SVO.*|.*stereo.*\.mp4$" "${DROID_RAW_GCS_PREFIX}" "${TARGET_DIR}"
