#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
BRIDGE_ROOT="${RAW_ROOT}/bridgedata_v2"
mkdir -p "${BRIDGE_ROOT}"
cd "${BRIDGE_ROOT}"
if [[ ! -f scripted_6_18.zip ]]; then
  wget -c https://rail.eecs.berkeley.edu/datasets/bridge_release/data/scripted_6_18.zip
fi
if [[ ! -d scripted_6_18 ]]; then
  unzip -q -n scripted_6_18.zip -d scripted_6_18
fi
