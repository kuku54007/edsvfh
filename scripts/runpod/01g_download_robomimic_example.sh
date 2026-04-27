#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ROBO_ROOT="${RAW_ROOT}/robomimic"
ROBO_REPO="${RAW_ROOT}/robomimic_repo"
if [[ ! -d "${ROBO_REPO}" ]]; then
  git clone https://github.com/ARISE-Initiative/robomimic.git "${ROBO_REPO}"
fi
cd "${ROBO_REPO}"
run_timed "01g_download_robomimic_example-pip" python -u -m pip install -U pip
run_timed "01g_download_robomimic_example-pip" python -u -m pip install -e .
python robomimic/scripts/download_datasets.py --tasks lift --dataset_types ph --hdf5_types low_dim --download_dir "${ROBO_ROOT}"
