#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ensure_dirs
mkdir -p "${DROID_FAILURE_WORK_ROOT}" \
         "${DROID_FAILURE_FRAMES_ROOT}" \
         "$(dirname "${DROID_FAILURE_MANIFEST_PATH}")" \
         "$(dirname "${DROID_FAILURE_PSEUDO_MANIFEST_PATH}")" \
         "$(dirname "${DROID_SUCCESS_BASELINE_PATH}")" \
         "$(dirname "${DROID_FAILURE_BUNDLE}")" \
         "$(dirname "${DROID_FAILURE_CKPT}")"
printf '[make-dirs] ready\n' >&2
