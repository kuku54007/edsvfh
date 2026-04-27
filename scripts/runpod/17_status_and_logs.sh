#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ensure_dirs
print_layout
if [[ $# -ge 1 ]]; then
  show_pid_status "$1" || true
fi
echo "--- tracked background processes ---"
while IFS= read -r pid_file; do
  [[ -n "${pid_file}" ]] || continue
  show_named_bg_status "${pid_file}"
  echo
done < <(bg_pid_files || true)
echo "--- latest logs ---"
for f in   "${DROID_FAILURE_MANIFEST_LOG}"   "${DROID_SUCCESS_BASELINE_LOG}"   "${DROID_FAILURE_PSEUDO_LABEL_LOG}"   "${DROID_FAILURE_CONVERT_LOG}"   "${DROID_FAILURE_TRAIN_LOG}"   "${DROID_FAILURE_PSEUDO_PIPELINE_LOG}"   "${FINO_MANIFEST_LOG}"   "${FINO_PSEUDO_LABEL_LOG}"   "${FINO_CONVERT_LOG}"   "${CONVERT_DROID_CURATED_LOG}"   "${TRAIN_DROID_CURATED_LOG}"   "${FINO_TRAIN_LOG}"   "${FINO_PSEUDO_PIPELINE_LOG}"   "${LOG_ROOT}/clean_rerun_fino_pseudo_onset.log"   "${CANARY_LOG}"   "${CANARY_SUMMARY_TXT}"   "${LARGER_HORIZONS_LOG}"   "${REPLAY_LOG}"   "${REPLAY_OUTPUT_JSON}"   "${ABLATION_LOG}"   "${ABLATION_OUTPUT_JSON}"   "${PAPER_PACK_LOG}"   "${LOG_ROOT}/full_paper_path_droid_failure.log"   "${LOG_ROOT}/smoke_test_full_paper_path_droid_failure.log"; do
  if [[ -f "$f" ]]; then
    echo "### $f"
    tail -n 20 "$f"
  fi
done
