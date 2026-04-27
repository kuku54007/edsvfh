#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
cd "${PROJECT_DIR}"

log_step "START 25_full_paper_path_droid_failure"
print_layout

bash scripts/runpod/00_preflight.sh
bash scripts/runpod/12_verify_project.sh
bash scripts/runpod/13_full_droid_then_droid_failure.sh

if should_force_rerun || [[ ! -f "${LARGER_HORIZONS_OUTPUT_BUNDLE}" ]]; then
  bash scripts/runpod/22_test_larger_horizons.sh
else
  log_step "SKIP larger-horizons training; found ${LARGER_HORIZONS_OUTPUT_BUNDLE}"
fi

if should_force_rerun || [[ ! -f "${REPLAY_OUTPUT_JSON}" ]]; then
  bash scripts/runpod/23_eval_replay_protocol.sh
else
  log_step "SKIP replay evaluation; found ${REPLAY_OUTPUT_JSON}"
fi

if should_force_rerun || [[ ! -f "${ABLATION_OUTPUT_JSON}" ]]; then
  bash scripts/runpod/24_eval_ablation_suite.sh
else
  log_step "SKIP ablation evaluation; found ${ABLATION_OUTPUT_JSON}"
fi

bash scripts/runpod/19_pack_paper_results.sh
log_step "END 25_full_paper_path_droid_failure"
