#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
BRIDGE_ROOT="${RAW_ROOT}/bridgedata_v2"
mkdir -p "${BRIDGE_ROOT}"
cd "${BRIDGE_ROOT}"
if [[ ! -f demos_8_17.zip ]]; then
  wget -c https://rail.eecs.berkeley.edu/datasets/bridge_release/data/demos_8_17.zip
fi
if [[ ! -d demos_8_17 ]]; then
  echo "Large archive detected; extraction may take a long time and consume substantial disk."
  unzip -q -n demos_8_17.zip -d demos_8_17
fi
