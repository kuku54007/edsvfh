#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ensure_dirs
cd "${PROJECT_DIR}"

python - <<'PY2'
import importlib
mods = ["torch", "torchvision"]
for name in mods:
    try:
        m = importlib.import_module(name)
        print(f"{name}={getattr(m, '__version__', 'unknown')}")
    except Exception as exc:
        raise SystemExit(f"[hf-runtime] {name} import failed: {exc}")
PY2

# Accept known-good RunPod base stacks without trying to mutate the template.
# We intentionally do NOT auto-upgrade torch / torchvision here.
set +e
python - <<'PY2'
import importlib
import json
import sys

def mm(v: str) -> str:
    parts = v.split('+', 1)[0].split('.')
    return '.'.join(parts[:2])

torch = importlib.import_module('torch')
tv = importlib.import_module('torchvision')
try:
    ta = importlib.import_module('torchaudio')
    ta_v = getattr(ta, '__version__', 'unknown')
except Exception:
    ta_v = None

t = getattr(torch, '__version__', 'unknown')
v = getattr(tv, '__version__', 'unknown')
accepted_pairs = {('2.4', '0.19'), ('2.7', '0.22'), ('2.8', '0.23'), ('2.9', '0.24')}
pair_ok = (mm(t), mm(v)) in accepted_pairs
# torchaudio is optional, but if present it should match torch major.minor
if ta_v is None:
    audio_ok = True
else:
    audio_ok = mm(ta_v) == mm(t)
# Blackwell (sm_120 / CC 12.0) requires PyTorch builds with CUDA 12.8 support.
blackwell = False
cap = None
try:
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        blackwell = tuple(cap) >= (12, 0)
except Exception:
    pass
if blackwell:
    pair_ok = pair_ok and (mm(t) in {'2.7','2.8','2.9'}) and ('+cu128' in t or '+cu129' in t or '+cu130' in t)
need_repair = not (pair_ok and audio_ok)
print(json.dumps({
    'torch': t,
    'torchvision': v,
    'torchaudio': ta_v,
    'pair_ok': pair_ok,
    'audio_ok': audio_ok,
    'blackwell': blackwell,
    'capability': cap,
    'need_repair': need_repair,
}, ensure_ascii=False))
sys.exit(3 if need_repair else 0)
PY2
rc=$?
set -e
if [[ $rc -eq 3 ]]; then
  echo "[hf-runtime] Unsupported or polluted torch stack detected."
  echo "[hf-runtime] Use one of these explicit repairs, then rerun bootstrap:"
  echo "  bash scripts/runpod/00_force_repair_torch_stack.sh cu124   # non-Blackwell only"
  echo "  bash scripts/runpod/00_force_repair_torch_stack.sh cu128   # required for Blackwell (sm_120)"
  exit 3
fi

# Keep the current compatible torch stack. Only install HF/runtime deps.
run_timed "00_repair_hf_runtime-pip" python -u -m pip install --upgrade "protobuf<7"
run_timed "00_repair_hf_runtime-pip" python -u -m pip install --upgrade   "transformers>=4.52,<5"   "safetensors>=0.4.3"   "huggingface-hub<1"

python - <<'PY2'
import importlib
mods = ["torch", "torchvision", "transformers", "safetensors", "huggingface_hub"]
for name in mods:
    m = importlib.import_module(name)
    print(f"{name}={getattr(m, '__version__', 'unknown')}")
PY2
