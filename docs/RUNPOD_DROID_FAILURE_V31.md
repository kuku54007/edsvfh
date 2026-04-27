# RunPod v31: DROID not-successful replacement for FINO

This is the recommended thesis-facing route for v31. It keeps the EDSV-FH method unchanged but replaces FINO/FAILURE with DROID not-successful trajectories as the failure-side dataset.

## Environment variables

Edit `scripts/runpod/runpod.env` if your paths differ:

```bash
export PROJECT_DIR=/workspace/edsvfh_public_v31
export DROID_RAW_ROOT=/workspace/data/raw/droid
export DROID_CURATED_ROOT=/workspace/data/converted/droid_curated
export DROID_FAILURE_RAW_ROOT=/workspace/data/raw/droid_raw/1.0.1
export DROID_FAILURE_WORK_ROOT=/workspace/data/raw/droid_failure
export DROID_FAILURE_FRAMES_ROOT=/workspace/data/raw/droid_failure/frames
export DROID_FAILURE_MANIFEST_PATH=/workspace/data/raw/droid_failure/droid_failure_manifest.jsonl
export DROID_FAILURE_PSEUDO_MANIFEST_PATH=/workspace/data/raw/droid_failure/droid_failure_manifest_pseudo_onset.jsonl
export DROID_FAILURE_CONVERTED_ROOT=/workspace/data/converted/droid_failure
export DROID_FAILURE_BUNDLE=/workspace/artifacts/droid_droid_failure_bundle.pkl
```

`DROID_FAILURE_RAW_ROOT` must point to a raw DROID tree containing the official raw episodes. In v31.1 the manifest scanner first checks metadata success/outcome fields and then falls back to path markers such as `failure`, `not_successful`, or `unsuccessful`. This is safer because the DROID paper states that success is logged in episode metadata, and official examples also show versioned raw paths such as `droid_raw/1.0.1/<site>/<success|failure>/...`.

## Helpful setup commands

Create the directory tree first:

```bash
bash scripts/runpod/00_make_dirs.sh
```

To download raw DROID for the failure-side route without stereo and SVO files:

```bash
bash scripts/runpod/01i_download_droid_raw_nonstereo.sh
```

## Full v31 route

```bash
source scripts/runpod/runpod.env
bash scripts/runpod/13_full_droid_then_droid_failure.sh
```

This performs preflight checks, DROID download/conversion/training, DROID not-successful manifest generation, pseudo-onset relabeling, failure-shard conversion, and DROID failure fine-tuning.

## Stage 2 only

Run this after you already have a DROID base bundle and DROID curated shards:

```bash
source scripts/runpod/runpod.env
bash scripts/runpod/07c_run_droid_failure_pseudo_onset_pipeline.sh
```

The Stage 2 wrapper executes:

```bash
bash scripts/runpod/06_generate_droid_failure_manifest.sh
bash scripts/runpod/07a_fit_droid_success_baseline.sh
bash scripts/runpod/07b_label_droid_failure_pseudo_onset.sh
bash scripts/runpod/07_convert_droid_failure.sh
bash scripts/runpod/08_finetune_droid_failure.sh
```

## Background execution

Use the background launchers if the browser session is unstable:

```bash
bash scripts/runpod/06b_launch_generate_droid_failure_manifest_bg.sh
bash scripts/runpod/07b_launch_label_droid_failure_pseudo_onset_bg.sh
bash scripts/runpod/07_launch_convert_droid_failure_bg.sh
bash scripts/runpod/08_launch_finetune_droid_failure_bg.sh
```

Check logs with:

```bash
bash scripts/runpod/17_status_and_logs.sh
```

## Extended horizons

After `DROID_FAILURE_BUNDLE` exists, run:

```bash
bash scripts/runpod/22_test_larger_horizons.sh
```

The default horizon set is `1,3,5,10,15,30,45,60`. The script can warm-start from the 1/3/5 bundle and train only newly added horizon heads when `LARGER_HORIZONS_FREEZE_EXISTING=1`.

## Thesis interpretation

Use the following wording in the paper:

> We use DROID success trajectories to learn a normal manipulation baseline and DROID not-successful trajectories as weakly labeled failure-side data. Because DROID does not provide dense frame-level failure-onset labels, we infer pseudo-onsets from the deviation between each not-successful trajectory and the DROID success baseline. The pseudo-onset labels are then converted into multi-window failure-horizon labels.

This preserves the EDSV-FH contribution: event-driven subgoal verification, pseudo-onset weak supervision, and action-chunk failure-horizon prediction.
