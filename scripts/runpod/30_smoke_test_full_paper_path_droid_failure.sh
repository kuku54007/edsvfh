#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
cd "${PROJECT_DIR}"

# Isolate smoke-test outputs so they do not overwrite formal artifacts.
export DROID_CURATED_ROOT="${CONVERTED_ROOT}/smoke_droid_curated"
export DROID_BASELINE_SHARD_DIR="${DROID_CURATED_ROOT}"
export BASE_BUNDLE="${ARTIFACT_ROOT}/smoke_droid_curated_bundle.pkl"
export DROID_SUCCESS_BASELINE_PATH="${ARTIFACT_ROOT}/smoke_droid_success_baseline.pkl"
export DROID_FAILURE_WORK_ROOT="${RAW_ROOT}/smoke_droid_failure"
export DROID_FAILURE_FRAMES_ROOT="${DROID_FAILURE_WORK_ROOT}/frames"
export DROID_FAILURE_MANIFEST_PATH="${DROID_FAILURE_WORK_ROOT}/droid_failure_manifest.jsonl"
export DROID_FAILURE_PSEUDO_MANIFEST_PATH="${DROID_FAILURE_WORK_ROOT}/droid_failure_manifest_pseudo_onset.jsonl"
export DROID_FAILURE_CONVERTED_ROOT="${CONVERTED_ROOT}/smoke_droid_failure"
export DROID_FAILURE_BUNDLE="${ARTIFACT_ROOT}/smoke_droid_droid_failure_bundle.pkl"
export DROID_FAILURE_PSEUDO_CKPT="${CHECKPOINT_ROOT}/smoke_droid_failure_pseudo_onset_ckpt.json"
export DROID_FAILURE_CONVERT_CKPT="${CHECKPOINT_ROOT}/smoke_droid_failure_convert_ckpt.json"
export DROID_FAILURE_CKPT="${CHECKPOINT_ROOT}/smoke_droid_failure_finetune_ckpt.pkl"
export DROID_FAILURE_MANIFEST_SCAN_CKPT="${CHECKPOINT_ROOT}/smoke_droid_failure_manifest_scan_ckpt.json"
export DROID_FAILURE_MANIFEST_LOG="${LOG_ROOT}/smoke_droid_failure_manifest.log"
export DROID_FAILURE_PSEUDO_LABEL_LOG="${LOG_ROOT}/smoke_label_droid_failure_pseudo_onset.log"
export DROID_FAILURE_CONVERT_LOG="${LOG_ROOT}/smoke_convert_droid_failure.log"
export DROID_FAILURE_TRAIN_LOG="${LOG_ROOT}/smoke_droid_failure_finetune.log"
export DROID_FAILURE_PSEUDO_PIPELINE_LOG="${LOG_ROOT}/smoke_droid_failure_pseudo_onset_pipeline.log"
export CONVERT_DROID_CURATED_LOG="${LOG_ROOT}/smoke_convert_droid_curated.log"
export TRAIN_DROID_CURATED_LOG="${LOG_ROOT}/smoke_train_droid_curated.log"
export DROID_CURATED_CKPT="${CHECKPOINT_ROOT}/smoke_droid_curated_train_ckpt.pkl"
export LARGER_HORIZONS="1,3,5,10,15"
export LARGER_HORIZONS_BASE_BUNDLE="${DROID_FAILURE_BUNDLE}"
export LARGER_HORIZONS_OUTPUT_BUNDLE="${ARTIFACT_ROOT}/smoke_droid_droid_failure_bundle_horizons_1x3x5x10x15.pkl"
export LARGER_HORIZONS_CKPT="${CHECKPOINT_ROOT}/smoke_droid_failure_larger_horizons_ckpt.pkl"
export LARGER_HORIZONS_LOG="${LOG_ROOT}/smoke_test_larger_horizons.log"
export REPLAY_BUNDLE="${LARGER_HORIZONS_OUTPUT_BUNDLE}"
export REPLAY_MAX_EPISODES="8"
export REPLAY_OUTPUT_JSON="${LOG_ROOT}/smoke_replay_protocol_metrics.json"
export REPLAY_OUTPUT_CSV="${LOG_ROOT}/smoke_replay_protocol_metrics.csv"
export REPLAY_LOG="${LOG_ROOT}/smoke_replay_protocol_eval.log"
export ABLATION_BUNDLES="main=${DROID_FAILURE_BUNDLE},extended=${LARGER_HORIZONS_OUTPUT_BUNDLE}"
export ABLATION_OUTPUT_JSON="${LOG_ROOT}/smoke_ablation_suite_metrics.json"
export ABLATION_OUTPUT_CSV="${LOG_ROOT}/smoke_ablation_suite_metrics.csv"
export ABLATION_LOG="${LOG_ROOT}/smoke_ablation_suite_eval.log"
export PAPER_PACK_PREFIX="paper_pack_smoke"
export PAPER_PACK_LOG="${LOG_ROOT}/smoke_paper_pack_results.log"
export PAPER_PACK_SKIP_EVAL="1"

export MAX_EPISODES="64"
export DROID_FAILURE_SCAN_MAX_EPISODES="256"
export DROID_FAILURE_MAX_EPISODES="24"
export DROID_FAILURE_MAX_FRAMES_PER_EPISODE="48"
export DROID_FAILURE_MANIFEST_SCAN_CKPT_EVERY="32"
export PSEUDO_ONSET_FIT_MAX_EPISODES="64"
export EPOCHS="1"
export LARGER_HORIZONS_EPOCHS="1"
export EDSVFH_FORCE_RERUN="1"

log_step "START 30_smoke_test_full_paper_path_droid_failure"
print_layout

bash scripts/runpod/00_preflight.sh
bash scripts/runpod/12_verify_project.sh
bash scripts/runpod/13_full_droid_then_droid_failure.sh
bash scripts/runpod/22_test_larger_horizons.sh
bash scripts/runpod/23_eval_replay_protocol.sh
bash scripts/runpod/24_eval_ablation_suite.sh
bash scripts/runpod/19_pack_paper_results.sh

log_step "END 30_smoke_test_full_paper_path_droid_failure"
