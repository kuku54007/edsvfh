# RunPod v33 from scratch: install -> smoke test -> formal run -> package

## 1. Unzip and enter the repo

```bash
cd /workspace
unzip -o edsvfh_public_v33.zip -d /workspace
cd /workspace/edsvfh_public_v33
```

## 2. Load the environment and create folders

```bash
source scripts/runpod/runpod.env
bash scripts/runpod/00_make_dirs.sh
```

## 3. Install dependencies once

```bash
bash scripts/runpod/00b_launch_bootstrap_bg.sh
bash scripts/runpod/17_status_and_logs.sh
tail -f /workspace/logs/bootstrap.log
```

Wait for `bootstrap.log` to finish successfully before training.

## 4. Confirm your DROID RLDS folder

Expected prepared TFDS layout:

```text
/workspace/data/raw/droid/1.0.1/
  dataset_info.json
  features.json
  droid_101-train.tfrecord-00000-...
```

If your data lives elsewhere:

```bash
export DROID_SOURCE=/workspace/data/raw/droid/1.0.1
export DROID_FAILURE_RLDS_ROOT=/workspace/data/raw/droid/1.0.1
```

## 5. Run the full smoke test first

```bash
bash scripts/runpod/30b_launch_smoke_test_full_paper_path_droid_failure_bg.sh
bash scripts/runpod/17_status_and_logs.sh
tail -f /workspace/logs/smoke_test_full_paper_path_droid_failure.log
```

The smoke test uses isolated `smoke_*` outputs, so it will not overwrite the formal run.

## 6. Formal full run in background

```bash
source scripts/runpod/runpod.env
export MAX_EPISODES=
export DROID_FAILURE_SCAN_MAX_EPISODES=
export DROID_FAILURE_MAX_EPISODES=
export DROID_FAILURE_MAX_FRAMES_PER_EPISODE=
export PSEUDO_ONSET_FIT_MAX_EPISODES=
export EPOCHS=3
export LARGER_HORIZONS_EPOCHS=3
export EDSVFH_FORCE_RERUN=0
bash scripts/runpod/25b_launch_full_paper_path_droid_failure_bg.sh
```

## 7. Resume behavior

The formal script is designed to skip completed artifacts:

- existing DROID curated shards -> skip `04_convert_droid_curated.sh`
- existing base bundle -> skip `05_train_droid_curated.sh`
- existing failure manifest -> skip manifest generation
- existing pseudo manifest -> skip pseudo-onset labeling
- existing DROID failure shards -> skip failure conversion
- existing main failure bundle -> skip main fine-tuning
- existing larger-horizon bundle -> skip `22_test_larger_horizons.sh`
- existing replay / ablation JSON -> skip those eval stages

If you really want to recompute a finished stage, either delete its outputs or set:

```bash
export EDSVFH_FORCE_RERUN=1
```

## 8. Package the finished run

```bash
bash scripts/runpod/19b_launch_pack_paper_results_bg.sh
tail -f /workspace/logs/paper_pack_results.log
```

Main outputs:

- `/workspace/artifacts/droid_curated_bundle.pkl`
- `/workspace/artifacts/droid_droid_failure_bundle.pkl`
- `/workspace/artifacts/droid_droid_failure_bundle_horizons_1x3x5x10x15x30x45x60.pkl`
- `/workspace/logs/replay_protocol_metrics.json`
- `/workspace/logs/ablation_suite_metrics.json`
- `/workspace/paper_pack_light_latest.tar.gz`
- `/workspace/paper_pack_full_latest.tar.gz`
