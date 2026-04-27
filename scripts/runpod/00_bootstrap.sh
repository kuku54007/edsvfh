#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
print_layout

cd "${PROJECT_DIR}"
run_timed "00_bootstrap-pip" python -u -m pip install --upgrade pip
run_timed "00_bootstrap-pip" python -u -m pip install --upgrade "protobuf<7"
run_timed "00_bootstrap-pip" python -u -m pip install -e ".[tfds,test]"
if [[ "${USE_HF}" == "1" || "${ENCODER}" != "fallback" ]]; then
  bash scripts/runpod/00_repair_hf_runtime.sh
fi
# Unit tests validate code contracts, not HF encoder downloads. Keep them
# deterministic and cheap even when the formal run uses ENCODER=siglip2_dinov2.
export ENCODER=fallback
export EDSVFH_ENCODER=fallback
export EDSVFH_DEVICE=cpu
export USE_HF=0
python -u -m pytest -q
