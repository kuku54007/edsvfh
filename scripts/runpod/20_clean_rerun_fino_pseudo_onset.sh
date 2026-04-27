#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
cd "${PROJECT_DIR}"

log_step "START clean FINO pseudo-onset rerun prep"
print_layout

rm -rf "${FINO_CONVERTED_ROOT}"
rm -f  "${FINO_CONVERT_CKPT}"
rm -f  "${FINO_CKPT}"
rm -f  "${FINO_PSEUDO_CKPT}"
rm -f  "${ARTIFACT_ROOT}/droid_fino_bundle.pkl"
rm -f  "${DROID_SUCCESS_BASELINE_PATH}"
rm -f  "${FINO_PSEUDO_MANIFEST_PATH}"
rm -f  "${FINO_PSEUDO_PIPELINE_LOG}"
rm -f  "${FINO_PSEUDO_LABEL_LOG}"
rm -f  "${FINO_CONVERT_LOG}"
rm -f  "${FINO_TRAIN_LOG}"
rm -f  "${LOG_ROOT}/07a_fit_droid_success_baseline.stdout.log"
rm -f  "${LOG_ROOT}/07b_label_fino_pseudo_onset.stdout.log"
rm -f  "${LOG_ROOT}/07_convert_fino.stdout.log"
rm -f  "${LOG_ROOT}/08_finetune_fino.stdout.log"
mkdir -p "${FINO_CONVERTED_ROOT}" "$(dirname "${FINO_CONVERT_CKPT}")" "$(dirname "${FINO_CKPT}")" "$(dirname "${FINO_PSEUDO_CKPT}")"

log_step "END clean FINO pseudo-onset rerun prep"

bash scripts/runpod/07c_run_fino_pseudo_onset_pipeline.sh
