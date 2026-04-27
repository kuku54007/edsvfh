#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
mkdir -p "${DROID_FAILURE_WORK_ROOT}" "${DROID_FAILURE_FRAMES_ROOT}" "${DROID_FAILURE_CONVERTED_ROOT}" "$(dirname "${DROID_SUCCESS_BASELINE_PATH}")" "$(dirname "${DROID_FAILURE_CKPT}")"
cd "${PROJECT_DIR}"

log_step "START DROID not-successful pseudo-onset pipeline"
print_layout

if should_force_rerun || [[ ! -f "${DROID_FAILURE_MANIFEST_PATH}" ]]; then
  log_step "Running DROID failure manifest generation"
  bash scripts/runpod/06_generate_droid_failure_manifest.sh
else
  log_step "SKIP manifest generation; found ${DROID_FAILURE_MANIFEST_PATH}"
fi

if [[ "${USE_PSEUDO_ONSET}" != "0" ]]; then
  if should_force_rerun || [[ ! -f "${DROID_SUCCESS_BASELINE_PATH}" ]]; then
    log_step "Running DROID success baseline fitting"
    bash scripts/runpod/07a_fit_droid_success_baseline.sh
  else
    log_step "SKIP baseline fitting; found ${DROID_SUCCESS_BASELINE_PATH}"
  fi
  if should_force_rerun || [[ ! -f "${DROID_FAILURE_PSEUDO_MANIFEST_PATH}" ]]; then
    log_step "Running DROID failure pseudo-onset labeling"
    bash scripts/runpod/07b_label_droid_failure_pseudo_onset.sh
  else
    log_step "SKIP pseudo-onset labeling; found ${DROID_FAILURE_PSEUDO_MANIFEST_PATH}"
  fi
fi

if should_force_rerun || ! dir_has_hdf5 "${DROID_FAILURE_CONVERTED_ROOT}"; then
  log_step "Running DROID failure shard conversion"
  bash scripts/runpod/07_convert_droid_failure.sh
else
  log_step "SKIP DROID failure conversion; HDF5 shards already exist under ${DROID_FAILURE_CONVERTED_ROOT}"
fi

if should_force_rerun || [[ ! -f "${DROID_FAILURE_BUNDLE}" ]]; then
  log_step "Running DROID failure fine-tuning"
  bash scripts/runpod/08_finetune_droid_failure.sh
else
  log_step "SKIP DROID failure fine-tuning; found ${DROID_FAILURE_BUNDLE}"
fi

log_step "END DROID not-successful pseudo-onset pipeline"
