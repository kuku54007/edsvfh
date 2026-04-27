#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ensure_dirs
cd "${PROJECT_DIR}"
bash scripts/runpod/00_ensure_runtime.sh
run_timed "12_verify_project-pip" python -u -m pip install --upgrade "protobuf<7"
# Verification must be deterministic and cheap. Do not inherit formal-run HF
# encoder settings into unit tests; formal conversion/training scripts still use
# the original ENCODER value in their own shell processes.
export EDSVFH_VERIFY_ORIGINAL_ENCODER="${ENCODER:-}"
export EDSVFH_VERIFY_ORIGINAL_USE_HF="${USE_HF:-}"
export ENCODER=fallback
export EDSVFH_ENCODER=fallback
export EDSVFH_DEVICE=cpu
export USE_HF=0
for t in tests/test_*.py; do
  run_timed "12_verify_project-pytest $(basename "$t")" python -u -m pytest -q "$t"
done
for f in scripts/runpod/*.sh; do
  bash -n "$f"
done
run_timed "12_verify_project" python -u -m edsvfh.cli catalog > /tmp/edsvfh_catalog.json
run_timed "12_verify_project" python -u -m edsvfh.cli make-fixture --output /tmp/edsvfh_fixture.hdf5
run_timed "12_verify_project" python -u -m edsvfh.cli train-fixture --fixture /tmp/edsvfh_fixture.hdf5 --output /tmp/edsvfh_fixture_bundle.pkl --encoder fallback > /tmp/edsvfh_train_fixture.json
run_timed "12_verify_project" python -u -m edsvfh.cli convert-mock-droid --output-dir /tmp/edsvfh_mock_droid --num-episodes 8 --episodes-per-shard 4 > /tmp/edsvfh_mock_droid.json
run_timed "12_verify_project" python -u -m edsvfh.cli train-sharded --shard-dir /tmp/edsvfh_mock_droid --output /tmp/edsvfh_mock_droid_bundle.pkl --encoder fallback > /tmp/edsvfh_mock_droid_train.json
run_timed "12_verify_project" python -u -m edsvfh.cli convert-mock-failure --root-dir /tmp/edsvfh_mock_failure_raw --output-dir /tmp/edsvfh_mock_failure --num-episodes 8 --episodes-per-shard 4 > /tmp/edsvfh_mock_failure.json
run_timed "12_verify_project" python -u -m edsvfh.cli fine-tune-fino --base-bundle /tmp/edsvfh_mock_droid_bundle.pkl --shard-dir /tmp/edsvfh_mock_failure --output /tmp/edsvfh_mock_fino_bundle.pkl --epochs 1 > /tmp/edsvfh_mock_fino_train.json
run_timed "12_verify_project" python -u -m edsvfh.cli generate-fino-manifest --root-dir /tmp/edsvfh_mock_failure_raw --output /tmp/edsvfh_mock_failure_manifest.jsonl > /tmp/edsvfh_manifest.json
run_timed "12_verify_project" python -u -m edsvfh.cli fit-droid-success-baseline --shard-dir /tmp/edsvfh_mock_droid --output /tmp/edsvfh_mock_success_baseline.pkl --encoder fallback --quantile 0.95 > /tmp/edsvfh_mock_success_baseline.json
run_timed "12_verify_project" python -u -m edsvfh.cli label-fino-pseudo-onset --manifest /tmp/edsvfh_mock_failure_raw/mock_fino_manifest.jsonl --baseline /tmp/edsvfh_mock_success_baseline.pkl --output /tmp/edsvfh_mock_failure_pseudo.jsonl --image-size 96 --encoder fallback > /tmp/edsvfh_mock_failure_pseudo.json
run_timed "12_verify_project" python -u -m edsvfh.cli convert-fino-manifest --manifest /tmp/edsvfh_mock_failure_pseudo.jsonl --output-dir /tmp/edsvfh_mock_failure_pseudo_shards --episodes-per-shard 4 --image-size 96 --prefer-pseudo-onset > /tmp/edsvfh_mock_failure_pseudo_shards.json
run_timed "12_verify_project" python -u -m edsvfh.cli rebuild-fino-pseudo-onset --droid-shard-dir /tmp/edsvfh_mock_droid --fino-manifest /tmp/edsvfh_mock_failure_raw/mock_fino_manifest.jsonl --base-bundle /tmp/edsvfh_mock_droid_bundle.pkl --baseline-output /tmp/edsvfh_rebuild_success_baseline.pkl --pseudo-manifest-output /tmp/edsvfh_rebuild_pseudo_manifest.jsonl --converted-output-dir /tmp/edsvfh_rebuild_failure_shards --output-bundle /tmp/edsvfh_rebuild_fino_bundle.pkl --encoder fallback --epochs 1 --quantile 0.95 > /tmp/edsvfh_rebuild_fino_pseudo.json
run_timed "12_verify_project" python -u -m edsvfh.cli label-droid-failure-pseudo-onset --manifest /tmp/edsvfh_mock_failure_raw/mock_fino_manifest.jsonl --baseline /tmp/edsvfh_mock_success_baseline.pkl --output /tmp/edsvfh_mock_droid_failure_pseudo.jsonl --image-size 96 --encoder fallback > /tmp/edsvfh_mock_droid_failure_pseudo.json
run_timed "12_verify_project" python -u -m edsvfh.cli convert-droid-failure-manifest --manifest /tmp/edsvfh_mock_droid_failure_pseudo.jsonl --output-dir /tmp/edsvfh_mock_droid_failure_pseudo_shards --episodes-per-shard 4 --image-size 96 --prefer-pseudo-onset > /tmp/edsvfh_mock_droid_failure_pseudo_shards.json
run_timed "12_verify_project" python -u -m edsvfh.cli fine-tune-droid-failure --base-bundle /tmp/edsvfh_mock_droid_bundle.pkl --shard-dir /tmp/edsvfh_mock_droid_failure_pseudo_shards --output /tmp/edsvfh_mock_droid_failure_bundle.pkl --encoder fallback --epochs 1 > /tmp/edsvfh_mock_droid_failure_train.json
echo "[12_verify_project] NOTE: /tmp/edsvfh_runpod_verify is a temporary verification sandbox, not your formal /workspace experiment root."
RUNPOD_VERIFY_ROOT=/tmp/edsvfh_runpod_verify
rm -rf "${RUNPOD_VERIFY_ROOT}"
mkdir -p "${RUNPOD_VERIFY_ROOT}"
export WORKSPACE_ROOT="${RUNPOD_VERIFY_ROOT}"
export PROJECT_DIR="$(pwd)"
export DATA_ROOT="${RUNPOD_VERIFY_ROOT}/data"
export RAW_ROOT="${DATA_ROOT}/raw"
export CONVERTED_ROOT="${DATA_ROOT}/converted"
export ARTIFACT_ROOT="${RUNPOD_VERIFY_ROOT}/artifacts"
export LOG_ROOT="${RUNPOD_VERIFY_ROOT}/logs"
export CACHE_ROOT="${RUNPOD_VERIFY_ROOT}/cache"
export CHECKPOINT_ROOT="${RUNPOD_VERIFY_ROOT}/checkpoints"
export DROID_RAW_ROOT="${RAW_ROOT}/droid"
export DROID_DEBUG_ROOT="${CONVERTED_ROOT}/droid_debug"
export DROID_CURATED_ROOT="${CONVERTED_ROOT}/droid_curated"
export FINO_RAW_ROOT="${RAW_ROOT}/fino"
export FINO_MANIFEST_PATH="${FINO_RAW_ROOT}/fino_manifest.jsonl"
export FINO_PSEUDO_MANIFEST_PATH="${FINO_RAW_ROOT}/fino_manifest_pseudo_onset.jsonl"
export FINO_CONVERTED_ROOT="${CONVERTED_ROOT}/fino"
export DROID_FAILURE_WORK_ROOT="${RAW_ROOT}/droid_failure"
export DROID_FAILURE_FRAMES_ROOT="${DROID_FAILURE_WORK_ROOT}/frames"
export DROID_FAILURE_MANIFEST_PATH="${DROID_FAILURE_WORK_ROOT}/droid_failure_manifest.jsonl"
export DROID_FAILURE_PSEUDO_MANIFEST_PATH="${DROID_FAILURE_WORK_ROOT}/droid_failure_manifest_pseudo_onset.jsonl"
export DROID_FAILURE_CONVERTED_ROOT="${CONVERTED_ROOT}/droid_failure"
export DROID_FAILURE_BUNDLE="${ARTIFACT_ROOT}/droid_droid_failure_bundle.pkl"
export DROID_FAILURE_PSEUDO_CKPT="${CHECKPOINT_ROOT}/droid_failure_pseudo_onset_ckpt.json"
export DROID_FAILURE_CONVERT_CKPT="${CHECKPOINT_ROOT}/droid_failure_convert_ckpt.json"
export DROID_FAILURE_CKPT="${CHECKPOINT_ROOT}/droid_failure_finetune_ckpt.pkl"
export DROID_SUCCESS_BASELINE_PATH="${ARTIFACT_ROOT}/droid_success_baseline.pkl"
export BASE_BUNDLE="${ARTIFACT_ROOT}/droid_curated_bundle.pkl"
export USE_HF=0
export ENCODER=fallback
export EPOCHS=1
ensure_dirs
python -u -m edsvfh.cli convert-mock-droid --output-dir "${DROID_CURATED_ROOT}" --num-episodes 8 --episodes-per-shard 4 > /tmp/edsvfh_runpod_mock_droid.json
python -u -m edsvfh.cli train-sharded --shard-dir "${DROID_CURATED_ROOT}" --output "${ARTIFACT_ROOT}/droid_curated_bundle.pkl" --encoder fallback --epochs 1 > /tmp/edsvfh_runpod_mock_droid_train.json
python -u -m edsvfh.cli convert-mock-failure --root-dir "${FINO_RAW_ROOT}" --output-dir "${RUNPOD_VERIFY_ROOT}/tmp_failure_unused" --num-episodes 8 --episodes-per-shard 4 > /tmp/edsvfh_runpod_mock_failure.json
run_timed "12_verify_project" bash scripts/runpod/07c_run_fino_pseudo_onset_pipeline.sh > /tmp/edsvfh_runpod_pipeline.json
cp "${FINO_RAW_ROOT}/mock_fino_manifest.jsonl" "${DROID_FAILURE_MANIFEST_PATH}"
run_timed "12_verify_project" bash scripts/runpod/07c_run_droid_failure_pseudo_onset_pipeline.sh > /tmp/edsvfh_runpod_droid_failure_pipeline.json
cat <<SUMMARY
Verification finished.
- pytest passed
- shell scripts syntax checked
- fixture train passed
- mock droid sharded train passed
- mock failure fine-tune passed
- manifest generation passed
- DROID success baseline fit passed
- FINO pseudo-onset labeling passed
- pseudo-onset reconversion passed
- one-shot pseudo-onset rebuild passed
- RunPod FINO pseudo-onset wrapper passed
- RunPod DROID failure pseudo-onset wrapper passed
SUMMARY
