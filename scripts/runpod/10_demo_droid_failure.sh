#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
cd "${PROJECT_DIR}"
DATASET=""
for split in eval calib train; do
  candidate=$(find "${DROID_FAILURE_CONVERTED_ROOT}/${split}" -maxdepth 1 -name '*.hdf5' 2>/dev/null | sort | head -n 1 || true)
  if [[ -n "${candidate}" ]]; then
    DATASET="${candidate}"
    break
  fi
done
if [[ -z "${DATASET}" ]]; then
  echo "No converted DROID failure shard found under ${DROID_FAILURE_CONVERTED_ROOT}." >&2
  exit 1
fi
python -m edsvfh.cli demo --bundle "${DROID_FAILURE_BUNDLE}" --dataset "${DATASET}" --episode-index 0
