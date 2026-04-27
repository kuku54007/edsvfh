#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
CALVIN_REPO="${RAW_ROOT}/calvin_repo"
CALVIN_ROOT="${RAW_ROOT}/calvin"
if [[ ! -d "${CALVIN_REPO}" ]]; then
  git clone --recurse-submodules https://github.com/mees/calvin.git "${CALVIN_REPO}"
fi
cd "${CALVIN_REPO}/dataset"
sh download_data.sh debug
mkdir -p "${CALVIN_ROOT}"
rsync -a . "${CALVIN_ROOT}/"
