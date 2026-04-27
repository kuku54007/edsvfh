#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
cd "${PROJECT_DIR}"

log_step "START 22_test_larger_horizons"
print_layout

BASE_FAILURE_BUNDLE_PATH="${LARGER_HORIZONS_BASE_BUNDLE}"
if [[ ! -f "${BASE_FAILURE_BUNDLE_PATH}" ]]; then
  echo "Missing DROID failure base bundle for warm-start larger horizons: ${BASE_FAILURE_BUNDLE_PATH}" >&2
  echo "Run the DROID failure pseudo-onset pipeline first:" >&2
  echo "  bash scripts/runpod/07c_launch_droid_failure_pseudo_onset_pipeline_bg.sh" >&2
  exit 1
fi

if ! dir_has_hdf5 "${DROID_FAILURE_CONVERTED_ROOT}/train"; then
  echo "No converted DROID failure train shards found under ${DROID_FAILURE_CONVERTED_ROOT}/train." >&2
  echo "Run the DROID failure pseudo-onset pipeline first:" >&2
  echo "  bash scripts/runpod/07c_launch_droid_failure_pseudo_onset_pipeline_bg.sh" >&2
  exit 1
fi

if [[ "${USE_HF}" == "1" || "${ENCODER}" != "fallback" ]]; then
  export EDSVFH_DISABLE_CUDNN="${EDSVFH_DISABLE_CUDNN:-1}"
  bash scripts/runpod/00_repair_hf_runtime.sh
fi

export EDSVFH_HORIZONS="${LARGER_HORIZONS}"
mkdir -p "$(dirname "${LARGER_HORIZONS_OUTPUT_BUNDLE}")" "$(dirname "${LARGER_HORIZONS_CKPT}")" "$(dirname "${LARGER_HORIZONS_LOG}")"

if [[ "${LARGER_HORIZONS_RESUME}" != "1" ]]; then
  rm -f "${LARGER_HORIZONS_OUTPUT_BUNDLE}" "${LARGER_HORIZONS_CKPT}"
fi

EFFECTIVE_UPDATE_SCALER="${LARGER_HORIZONS_UPDATE_SCALER}"
if [[ "${LARGER_HORIZONS_FREEZE_EXISTING}" == "1" && "${EFFECTIVE_UPDATE_SCALER}" != "0" ]]; then
  echo "[22] LARGER_HORIZONS_FREEZE_EXISTING=1; forcing LARGER_HORIZONS_UPDATE_SCALER=0 to preserve existing heads." >&2
  EFFECTIVE_UPDATE_SCALER=0
fi

CMD=(
  python -u -m edsvfh.cli fine-tune-droid-failure
  --base-bundle "${BASE_FAILURE_BUNDLE_PATH}"
  --shard-dir "${DROID_FAILURE_CONVERTED_ROOT}"
  --output "${LARGER_HORIZONS_OUTPUT_BUNDLE}"
  --encoder "${ENCODER}"
  --epochs "${LARGER_HORIZONS_EPOCHS}"
  --horizons "${LARGER_HORIZONS}"
  --checkpoint "${LARGER_HORIZONS_CKPT}"
  --checkpoint-every 1
)
if [[ "${LARGER_HORIZONS_RESUME}" != "1" ]]; then
  CMD+=(--no-resume)
fi
if [[ "${LARGER_HORIZONS_FREEZE_EXISTING}" == "1" ]]; then
  CMD+=(--freeze-existing-horizons)
fi
if [[ "${EFFECTIVE_UPDATE_SCALER}" != "0" ]]; then
  CMD+=(--update-scaler)
fi

run_timed "22_test_larger_horizons" "${CMD[@]}"

python - <<'PY2'
import json
import os
from pathlib import Path
from edsvfh.models import VerifierBundle

path = Path(os.environ['LARGER_HORIZONS_OUTPUT_BUNDLE'])
if not path.exists():
    raise SystemExit(f"Missing output bundle: {path}")
bundle = VerifierBundle.load(path)
print(json.dumps({
    "larger_horizons_output_bundle": str(path),
    "larger_horizons_base_bundle": os.environ.get('LARGER_HORIZONS_BASE_BUNDLE'),
    "freeze_existing_horizons": os.environ.get('LARGER_HORIZONS_FREEZE_EXISTING'),
    "horizons": list(bundle.horizons),
    "input_dim": bundle.input_dim,
    "metadata": bundle.metadata,
}, indent=2, ensure_ascii=False))
PY2

log_step "END 22_test_larger_horizons"
