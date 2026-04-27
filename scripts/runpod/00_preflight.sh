#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ensure_dirs
print_layout
python --version
run_timed "00_preflight-pip" python -u -m pip --version
bash scripts/runpod/00_ensure_runtime.sh
if [[ "${USE_HF}" == "1" || "${ENCODER}" != "fallback" ]]; then
  bash scripts/runpod/00_repair_hf_runtime.sh
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found; GPU may be unavailable"
fi
python - <<'PY'
from pathlib import Path
import os
required = [
    Path(os.environ['WORKSPACE_ROOT']),
    Path(os.environ['PROJECT_DIR']),
]
for p in required:
    print(f"[check] {p} exists={p.exists()}")
PY
