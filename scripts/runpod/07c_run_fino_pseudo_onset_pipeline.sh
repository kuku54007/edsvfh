#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
mkdir -p "$(dirname "${FINO_MANIFEST_PATH}")"          "$(dirname "${FINO_PSEUDO_MANIFEST_PATH}")"          "${FINO_CONVERTED_ROOT}"          "$(dirname "${DROID_SUCCESS_BASELINE_PATH}")"          "$(dirname "${FINO_CKPT}")"
cd "${PROJECT_DIR}"

log_step "START FINO pseudo-onset pipeline"
print_layout

if [[ ! -f "${FINO_MANIFEST_PATH}" ]]; then
  log_step "FINO manifest missing; generating ${FINO_MANIFEST_PATH}"
  bash scripts/runpod/06_generate_fino_manifest.sh
fi

if [[ "${USE_PSEUDO_ONSET}" != "0" ]]; then
  bash scripts/runpod/07a_fit_droid_success_baseline.sh
  bash scripts/runpod/07b_label_fino_pseudo_onset.sh
fi

bash scripts/runpod/07_convert_fino.sh
bash scripts/runpod/08_finetune_fino.sh

log_step "END FINO pseudo-onset pipeline"
