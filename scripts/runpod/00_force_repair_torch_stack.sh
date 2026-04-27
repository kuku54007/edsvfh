#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ensure_dirs
cd "${PROJECT_DIR}"
profile="${1:-cu124}"
case "$profile" in
  cu124)
    index_url="https://download.pytorch.org/whl/cu124"
    torch_v="2.4.1"
    tv_v="0.19.1"
    ta_v="2.4.1"
    ;;
  cu128)
    index_url="https://download.pytorch.org/whl/cu128"
    torch_v="2.7.1"
    tv_v="0.22.1"
    ta_v="2.7.1"
    ;;
  *)
    echo "Usage: bash scripts/runpod/00_force_repair_torch_stack.sh [cu124|cu128]" >&2
    exit 2
    ;;
esac

run_timed "00_force_repair_torch_stack-uninstall" python -u -m pip uninstall -y torch torchvision torchaudio triton || true
run_timed "00_force_repair_torch_stack-install" python -u -m pip install --index-url "$index_url"   "torch==${torch_v}" "torchvision==${tv_v}" "torchaudio==${ta_v}"
run_timed "00_force_repair_torch_stack-hf" bash scripts/runpod/00_repair_hf_runtime.sh
