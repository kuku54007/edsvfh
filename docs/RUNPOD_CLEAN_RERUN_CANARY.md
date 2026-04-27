# RunPod clean-rerun canary

Use this before a full paid rerun.

## Frontground

```bash
cd /workspace/edsvfh_public_v30
source scripts/runpod/runpod.env
bash scripts/runpod/21_validate_clean_rerun_canary.sh
```

## Background

```bash
cd /workspace/edsvfh_public_v30
source scripts/runpod/runpod.env
bash scripts/runpod/21b_launch_validate_clean_rerun_canary_bg.sh
```

## What it does

1. Runs three targeted pytest checks for:
   - manifest-aware FINO reconversion
   - low-confidence pseudo-onset fallback keeping the original onset
   - end-to-end pseudo-onset rebuild
2. Runs a cheap clean rerun with:
   - `CANARY_PSEUDO_ONSET_FIT_MAX_EPISODES`
   - `CANARY_EPOCHS`
3. Verifies that:
   - pseudo manifest exists
   - not all failure episodes collapse to onset 0
   - convert checkpoint points at the pseudo manifest
   - convert checkpoint contains `manifest_sha256` and `resume_signature`
   - fine-tuned bundle exists

## Outputs

- `${CANARY_LOG}`
- `${CANARY_SUMMARY_JSON}`
- `${CANARY_SUMMARY_TXT}`
