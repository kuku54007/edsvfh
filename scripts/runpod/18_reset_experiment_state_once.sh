#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs

if [[ "${1:-}" != "--yes" ]]; then
  cat <<EOF
This script deletes generated training state once so you can rerun the project from scratch.
Raw downloads are preserved.

It will remove:
- ${DROID_DEBUG_ROOT}
- ${DROID_CURATED_ROOT}
- ${FINO_CONVERTED_ROOT}
- ${CHECKPOINT_ROOT}
- ${LOG_ROOT}
- generated bundle outputs under ${ARTIFACT_ROOT}, including droid_success_baseline*.pkl

It will preserve:
- ${RAW_ROOT}
- downloaded DROID / FINO raw data
- source code and docs

Run again with:
  bash scripts/runpod/18_reset_experiment_state_once.sh --yes
EOF
  exit 0
fi

log_step "START one-time experiment state reset"
rm -rf "${DROID_DEBUG_ROOT}" "${DROID_CURATED_ROOT}" "${FINO_CONVERTED_ROOT}" "${CHECKPOINT_ROOT}" "${LOG_ROOT}"
mkdir -p "${DROID_DEBUG_ROOT}" "${DROID_CURATED_ROOT}" "${FINO_CONVERTED_ROOT}" "${CHECKPOINT_ROOT}" "${LOG_ROOT}"
find "${ARTIFACT_ROOT}" -maxdepth 1 -type f \( \
  -name 'droid_debug_bundle*.pkl' -o \
  -name 'droid_curated_bundle*.pkl' -o \
  -name 'droid_fino_bundle*.pkl' -o \
  -name 'droid_success_baseline*.pkl' -o \
  -name '*.train_ckpt.pkl' -o \
  -name '*validation*.json' -o \
  -name '*summary*.json' \
\) -print -delete || true
log_step "END one-time experiment state reset | raw downloads preserved"
