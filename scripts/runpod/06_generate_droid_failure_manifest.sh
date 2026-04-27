#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
mkdir -p "$(dirname "${DROID_FAILURE_MANIFEST_PATH}")" "${DROID_FAILURE_FRAMES_ROOT}"
cd "${PROJECT_DIR}"

MODE="${DROID_FAILURE_SOURCE_MODE:-rlds}"
CAMERA_PREF="${DROID_FAILURE_CAMERA_PREFERENCE:-exterior_image_1_left,exterior_image_2_left,wrist_image_left,ext1,ext2,wrist}"

if [[ -n "${MAX_EPISODES:-}" && -z "${DROID_FAILURE_SCAN_MAX_EPISODES:-}" && -z "${DROID_FAILURE_MAX_EPISODES:-}" ]]; then
  log_step "WARNING: MAX_EPISODES=${MAX_EPISODES} is set for DROID curated conversion, but DROID failure scan has no limit. This will scan the full RLDS split."
fi

if [[ "${MODE}" == "raw" ]]; then
  CMD=(
    python -u -m edsvfh.cli generate-droid-failure-manifest
    --root-dir "${DROID_FAILURE_RAW_ROOT}"
    --output "${DROID_FAILURE_MANIFEST_PATH}"
    --frames-root "${DROID_FAILURE_FRAMES_ROOT}"
    --image-size "${IMAGE_SIZE}"
    --frame-stride "${DROID_FAILURE_FRAME_STRIDE}"
    --camera-preference "${CAMERA_PREF}"
  )
elif [[ "${MODE}" == "rlds" || "${MODE}" == "tfds" ]]; then
  SOURCE="${DROID_FAILURE_RLDS_ROOT:-${DROID_SOURCE:-${DROID_RAW_ROOT}/1.0.1}}"
  CMD=(
    python -u -m edsvfh.cli generate-droid-rlds-failure-manifest
    --source "${SOURCE}"
    --output "${DROID_FAILURE_MANIFEST_PATH}"
    --frames-root "${DROID_FAILURE_FRAMES_ROOT}"
    --split "${DROID_FAILURE_RLDS_SPLIT:-train}"
    --image-size "${IMAGE_SIZE}"
    --frame-stride "${DROID_FAILURE_FRAME_STRIDE}"
    --camera-preference "${CAMERA_PREF}"
  )
  if [[ -n "${DROID_FAILURE_SCAN_MAX_EPISODES:-}" ]]; then
    CMD+=(--scan-max-episodes "${DROID_FAILURE_SCAN_MAX_EPISODES}")
  fi
  if [[ -n "${DROID_FAILURE_MANIFEST_SCAN_CKPT:-}" ]]; then
    CMD+=(--checkpoint "${DROID_FAILURE_MANIFEST_SCAN_CKPT}")
  fi
  if [[ -n "${DROID_FAILURE_MANIFEST_SCAN_CKPT_EVERY:-}" ]]; then
    CMD+=(--checkpoint-every "${DROID_FAILURE_MANIFEST_SCAN_CKPT_EVERY}")
  fi
  if [[ "${DROID_FAILURE_MANIFEST_SCAN_RESUME:-1}" == "0" ]]; then
    CMD+=(--no-resume)
  fi
else
  echo "Unsupported DROID_FAILURE_SOURCE_MODE=${MODE}; expected rlds or raw." >&2
  exit 2
fi

if [[ -n "${DROID_FAILURE_MAX_EPISODES:-}" ]]; then
  CMD+=(--max-episodes "${DROID_FAILURE_MAX_EPISODES}")
fi
if [[ -n "${DROID_FAILURE_MAX_FRAMES_PER_EPISODE:-}" ]]; then
  CMD+=(--max-frames-per-episode "${DROID_FAILURE_MAX_FRAMES_PER_EPISODE}")
fi
if [[ "${DROID_FAILURE_OVERWRITE_FRAMES:-0}" == "1" ]]; then
  CMD+=(--overwrite-frames)
fi
run_timed "06_generate_droid_failure_manifest_${MODE}" "${CMD[@]}"
