#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
cd "${PROJECT_DIR}"

NEED_INSTALL=0
set +e
CHECK_OUTPUT="$(python - <<'PY' 2>&1
import importlib.util
required = [
    ('pytest', 'pytest'),
    ('fastapi', 'fastapi'),
    ('pydantic', 'pydantic'),
    ('h5py', 'h5py'),
    ('cv2', 'opencv-python'),
    ('sklearn', 'scikit-learn'),
    ('PIL', 'pillow'),
    ('tensorflow_datasets', 'tensorflow-datasets'),
]
missing = [pkg for mod, pkg in required if importlib.util.find_spec(mod) is None]
if missing:
    print('missing: ' + ', '.join(missing))
    raise SystemExit(2)
print('runtime dependencies present')
PY
)"
STATUS=$?
set -e
if [[ "${STATUS}" != "0" ]]; then
  NEED_INSTALL=1
fi

echo "[ensure-runtime] ${CHECK_OUTPUT}"

if [[ "${NEED_INSTALL}" == "1" ]]; then
  run_timed "00_ensure_runtime-pip" python -u -m pip install --upgrade pip
  run_timed "00_ensure_runtime-pip" python -u -m pip install --upgrade "protobuf<7"
  run_timed "00_ensure_runtime-pip" python -u -m pip install -e ".[tfds,test]"
fi
