#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
LIBERO_ROOT="${RAW_ROOT}/libero"
LIBERO_REPO="${RAW_ROOT}/libero_repo"
LIBERO_DATASET="${LIBERO_DATASET:-libero_100}"
if [[ ! -d "${LIBERO_REPO}" ]]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_REPO}"
fi
cd "${LIBERO_REPO}"
run_timed "01d_download_libero-pip" python -u -m pip install -U pip
python benchmark_scripts/download_libero_datasets.py --datasets "${LIBERO_DATASET}" --use-huggingface
mkdir -p "${LIBERO_ROOT}"
if [[ -d datasets ]]; then
  rsync -a datasets/ "${LIBERO_ROOT}/"
fi
