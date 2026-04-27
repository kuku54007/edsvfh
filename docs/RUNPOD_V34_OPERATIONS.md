# RUNPOD_V34_OPERATIONS

## 1. Install / update the project

```bash
cd /workspace
unzip -o edsvfh_public_v34.zip -d /workspace
cd /workspace/edsvfh_public_v34
source scripts/runpod/runpod.env
bash scripts/runpod/00_make_dirs.sh
bash scripts/runpod/00b_launch_bootstrap_bg.sh
```

Check bootstrap:

```bash
bash scripts/runpod/17_status_and_logs.sh
```

## 2. Smoke test first (cheap full-path rehearsal)

```bash
cd /workspace/edsvfh_public_v34
source scripts/runpod/runpod.env
bash scripts/runpod/30b_launch_smoke_test_full_paper_path_droid_failure_bg.sh
```

Smoke outputs are isolated:

- `DROID_CURATED_ROOT=/workspace/data/converted/smoke_droid_curated`
- `BASE_BUNDLE=/workspace/artifacts/smoke_droid_curated_bundle.pkl`
- `DROID_FAILURE_WORK_ROOT=/workspace/data/raw/smoke_droid_failure`
- `DROID_FAILURE_CONVERTED_ROOT=/workspace/data/converted/smoke_droid_failure`
- `DROID_FAILURE_BUNDLE=/workspace/artifacts/smoke_droid_droid_failure_bundle.pkl`
- `LARGER_HORIZONS_OUTPUT_BUNDLE=/workspace/artifacts/smoke_droid_droid_failure_bundle_horizons_1x3x5x10x15.pkl`

Smoke parent log:

```bash
/workspace/logs/smoke_test_full_paper_path_droid_failure.log
```

## 3. Formal full run in background

```bash
cd /workspace/edsvfh_public_v34
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

Formal parent log:

```bash
/workspace/logs/full_paper_path_droid_failure.log
```

## 4. Monitor background progress

General status:

```bash
bash scripts/runpod/17_status_and_logs.sh
```

Follow the formal parent log only:

```bash
tail -f /workspace/logs/full_paper_path_droid_failure.log
```

Follow the smoke parent log only:

```bash
tail -f /workspace/logs/smoke_test_full_paper_path_droid_failure.log
```

Show stage markers only:

```bash
grep -E 'START |END   |SKIP |Running ' /workspace/logs/full_paper_path_droid_failure.log | tail -n 80
```

## 5. Main formal paths

After `source scripts/runpod/runpod.env`, the formal defaults are:

- `PROJECT_DIR=/workspace/edsvfh_public_v34`
- `WORKSPACE_ROOT=/workspace`
- `DROID_SOURCE=/workspace/data/raw/droid/1.0.1`
- `DROID_CURATED_ROOT=/workspace/data/converted/droid_curated`
- `BASE_BUNDLE=/workspace/artifacts/droid_curated_bundle.pkl`
- `DROID_SUCCESS_BASELINE_PATH=/workspace/artifacts/droid_success_baseline.pkl`
- `DROID_FAILURE_WORK_ROOT=/workspace/data/raw/droid_failure`
- `DROID_FAILURE_MANIFEST_PATH=/workspace/data/raw/droid_failure/droid_failure_manifest.jsonl`
- `DROID_FAILURE_PSEUDO_MANIFEST_PATH=/workspace/data/raw/droid_failure/droid_failure_manifest_pseudo_onset.jsonl`
- `DROID_FAILURE_CONVERTED_ROOT=/workspace/data/converted/droid_failure`
- `DROID_FAILURE_BUNDLE=/workspace/artifacts/droid_droid_failure_bundle.pkl`
- `LARGER_HORIZONS_OUTPUT_BUNDLE=/workspace/artifacts/droid_droid_failure_bundle_horizons_1x3x5x10x15x30x45x60.pkl`
- `REPLAY_OUTPUT_JSON=/workspace/logs/replay_protocol_metrics.json`
- `ABLATION_OUTPUT_JSON=/workspace/logs/ablation_suite_metrics.json`
- `PAPER_PACK_OUTPUT_ROOT=/workspace`
- `PAPER_PACK_PREFIX=paper_pack`

## 6. Pack results after the formal run

```bash
cd /workspace/edsvfh_public_v34
source scripts/runpod/runpod.env
bash scripts/runpod/19b_launch_pack_paper_results_bg.sh
```

Expected pack outputs:

- `/workspace/paper_pack_light_latest.tar.gz`
- `/workspace/paper_pack_full_latest.tar.gz`

## 7. Files to provide for Chapter 4 writing

Best option:

- `/workspace/paper_pack_light_latest.tar.gz`
- `/workspace/paper_pack_full_latest.tar.gz`

Minimum useful set:

- `/workspace/logs/convert_droid_curated.log`
- `/workspace/logs/train_droid_curated.log`
- `/workspace/logs/droid_failure_manifest.log`
- `/workspace/logs/label_droid_failure_pseudo_onset.log`
- `/workspace/logs/convert_droid_failure.log`
- `/workspace/logs/droid_failure_finetune.log`
- `/workspace/logs/test_droid_failure_larger_horizons_1x3x5x10x15x30x45x60.log`
- `/workspace/logs/replay_protocol_metrics.json`
- `/workspace/logs/ablation_suite_metrics.json`
- `/workspace/artifacts/droid_curated_bundle.pkl`
- `/workspace/artifacts/droid_droid_failure_bundle.pkl`
- `/workspace/artifacts/droid_droid_failure_bundle_horizons_1x3x5x10x15x30x45x60.pkl`
