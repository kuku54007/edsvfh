#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ensure_dirs
cd "${PROJECT_DIR}"

log_step "START 24_eval_ablation_suite"
print_layout

export EDSVFH_HF_LOCAL_ONLY="${EDSVFH_HF_LOCAL_ONLY:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if ! dir_has_hdf5 "${DROID_FAILURE_CONVERTED_ROOT}/eval"; then
  echo "Missing DROID failure eval shards under ${DROID_FAILURE_CONVERTED_ROOT}/eval" >&2
  exit 1
fi

CMD=(
  python -u -m edsvfh.eval_protocols ablation
  --shard-dir "${DROID_FAILURE_CONVERTED_ROOT}"
  --variants "${ABLATION_VARIANTS}"
  --ece-bins "${ABLATION_ECE_BINS}"
  --output-json "${ABLATION_OUTPUT_JSON}"
  --output-csv "${ABLATION_OUTPUT_CSV}"
)
IFS=',' read -ra BUNDLE_SPECS <<< "${ABLATION_BUNDLES}"
for spec in "${BUNDLE_SPECS[@]}"; do
  spec="$(echo "${spec}" | xargs)"
  [[ -n "${spec}" ]] || continue
  CMD+=(--bundle "${spec}")
done

run_timed "24_eval_ablation_suite" "${CMD[@]}"
log_step "END 24_eval_ablation_suite"
