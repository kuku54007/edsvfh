#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
mkdir -p "${FINO_RAW_ROOT}"
cat <<MSG
FINO 需要手動下載，因為官方入口位於 GitHub README 的 Download Data / Download Annotations 連結。
請打開： https://github.com/ardai/fino-net
把資料解壓到： ${FINO_RAW_ROOT}
完成後再執行：
  bash scripts/runpod/06_generate_fino_manifest.sh
  bash scripts/runpod/07_convert_fino.sh
  bash scripts/runpod/08_finetune_fino.sh
MSG
