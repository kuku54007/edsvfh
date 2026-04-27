# RunPod larger-horizon experiment (v29)

v30 retains the warm-start larger-horizon test to the intended comparison:

```text
Keep existing 1/3/5 FINO fine-tuned horizon heads.
Train only the newly added 10/15 heads.
```

This avoids the v28 issue where `22_test_larger_horizons.sh` reinitialized all five heads, including the already useful 1/3/5 heads.

## Required inputs

The clean pseudo-onset rerun must already have completed. These files should exist:

```bash
/workspace/artifacts/droid_fino_bundle.pkl
/workspace/data/converted/fino/train
/workspace/data/converted/fino/calib
/workspace/data/converted/fino/eval
```

## Run

```bash
cd /workspace/edsvfh_public_v30
source scripts/runpod/runpod.env
bash scripts/runpod/22b_launch_test_larger_horizons_bg.sh
```

Watch progress:

```bash
bash scripts/runpod/17_status_and_logs.sh
tail -f /workspace/logs/test_larger_horizons_1x3x5x10x15.log
```

## Defaults

```bash
export LARGER_HORIZONS=1,3,5,10,15
export LARGER_HORIZONS_BASE_BUNDLE=/workspace/artifacts/droid_fino_bundle.pkl
export LARGER_HORIZONS_FREEZE_EXISTING=1
export LARGER_HORIZONS_UPDATE_SCALER=0
export LARGER_HORIZONS_OUTPUT_BUNDLE=/workspace/artifacts/droid_fino_bundle_horizons_1x3x5x10x15.pkl
```

`LARGER_HORIZONS_UPDATE_SCALER` is intentionally `0`. Frozen 1/3/5 heads must keep the scaler they were trained with.

## Expected metadata

The output bundle should report strategies like:

```text
kept_existing_frozen
kept_existing_frozen
kept_existing_frozen
reinitialized_sgd
reinitialized_sgd
```

This means 1/3/5 were preserved and only 10/15 were added.
