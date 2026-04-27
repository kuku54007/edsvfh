# EDSV-FH Public Dataset Project

This project is a **validated reference implementation** of the Chapter 2 architecture:

- Event Watcher
- Event Memory + Context Builder
- Subgoal State Estimator
- Predictive Failure Horizon Estimator
- Calibration + Decision Layer
- Online termination semantics (`shield` / `abstain` stop autonomy until reset)
- FastAPI integration
- public-dataset adapters
- DROID RLDS sharded pretraining path
- v31 default: DROID not-successful failure fine-tuning path with DROID-success pseudo-onset relabeling
- legacy FINO-style route retained for backward compatibility

The code is organized around one practical rule:

> **Convert large public datasets into robomimic-compatible HDF5 shards, then train / fine-tune incrementally.**

That keeps the Chapter 2 runtime stable while avoiding the need to load the full dataset into memory.

## v31 thesis-facing change: DROID not-successful replaces FINO

v31 changes the paper-oriented Stage 2 path from **DROID success -> FINO failure-side adaptation** to **DROID success -> DROID not-successful failure-side adaptation**. The method remains aligned with EDSV-FH because the core supervision target is still pseudo-onset and multi-window failure horizon prediction. The difference is the source of failure episodes: instead of a cross-domain FINO/FAILURE target domain, v31 uses the DROID release's not-successful trajectories as the failure-side dataset.

The resulting thesis phrasing should become: DROID success trajectories are used for base representation learning and success-normal baseline fitting; DROID not-successful trajectories are used for pseudo-onset relabeling, failure-horizon fine-tuning, calibration, and evaluation. This produces a single-dataset, outcome-split protocol and avoids the DROID-to-FINO robot/domain mismatch.

Primary RunPod route:

```bash
bash scripts/runpod/13_full_droid_then_droid_failure.sh
```

Primary one-shot Stage 2 route after DROID base training:

```bash
bash scripts/runpod/07c_run_droid_failure_pseudo_onset_pipeline.sh
```

FINO-related commands and scripts are still packaged as legacy compatibility helpers, but they are no longer the recommended thesis path for v31.


## Chapter 2 alignment

A section-by-section mapping from the thesis methodology to the implementation is included in:

- `docs/第二章_對齊說明.md`

This mapping explicitly ties the codebase to:
- Section 2.4 Event Watcher
- Section 2.5 Event Memory + Context Builder
- Section 2.6 Subgoal State Estimator
- Section 2.7 Predictive Failure Horizon
- Section 2.8 Calibration + Decision Layer
- Section 2.9 two-stage training (`DROID success -> DROID not-successful`; legacy `DROID -> FINO` retained)
- Section 2.10 online inference with termination semantics

## Remaining-time progress reporting

To reduce wasted GPU rental time, the long-running commands now emit **elapsed time, throughput, and remaining time** to `stderr` while keeping the final structured JSON result on `stdout`. The remaining-time estimate is implemented for:

- `convert-droid`
- `train-sharded`
- `generate-droid-failure-manifest`
- `fit-droid-success-baseline`
- `label-droid-failure-pseudo-onset`
- `convert-droid-failure-manifest`
- `fine-tune-droid-failure`
- `rebuild-droid-failure-pseudo-onset`
- legacy: `generate-fino-manifest`, `label-fino-pseudo-onset`, `convert-fino-manifest`, `fine-tune-fino`, `rebuild-fino-pseudo-onset`

If you need silent execution for automation, each relevant CLI command accepts `--no-progress`.

When a job is resumed, the estimator now computes throughput from **new work completed in the current run**, not from the already-finished historical count. That prevents the misleading `remaining 00:00` behavior that appeared immediately after resume.


> **重要相容性提醒（RunPod / Python 3.12）：** 若你在 `convert-droid` 遇到 `FieldDescriptor` 沒有 `label` 的錯誤，這通常是 `protobuf>=7` 與 TFDS prepared builder 之間的相容性問題。此專案已加入 runtime shim，且 `tfds` 依賴現在固定為 `protobuf<7` 以降低風險。


## One-time reset for a full rerun

If you want to rerun the whole Chapter 2 experiment from scratch **without re-downloading raw DROID / FINO data**, use:

```bash
bash scripts/runpod/18_reset_experiment_state_once.sh --yes
```

This is a **manual cleanup command**. It is **not** called automatically by any training script. It deletes generated converted shards, checkpoints, logs, and top-level generated bundles, while preserving `data/raw`.

## Checkpoint / resume and background-safe RunPod usage

This version adds **periodic checkpoints** and **resume** support for the long Chapter 2 paths:
- automatic quarantine of **corrupt trailing HDF5 shards** left by interrupted `convert-droid` runs

- `convert-droid`
- `train-sharded`
- `generate-fino-manifest`
- `label-fino-pseudo-onset`
- `convert-fino-manifest`
- `fine-tune-fino`

Default RunPod checkpoint files:

- `/workspace/checkpoints/droid_debug_train_ckpt.pkl`
- `/workspace/checkpoints/droid_curated_train_ckpt.pkl`
- `/workspace/checkpoints/fino_finetune_ckpt.pkl`
- `/workspace/data/converted/droid_* /.convert_state.json`
- `/workspace/data/converted/fino /.convert_fino_state.json`

If a RunPod job is interrupted, rerunning the same script will **resume from the latest checkpoint** by default. If the interruption leaves the active DROID shard unreadable, the converter will quarantine the corrupt trailing shard automatically and resume from the last known-good point.

For safer long jobs over unstable browser sessions, the package also includes background launchers:

- `scripts/runpod/14_launch_convert_droid_curated_bg.sh`
- `scripts/runpod/15_launch_train_droid_curated_bg.sh`
- `scripts/runpod/16_launch_fino_finetune_bg.sh`
- `scripts/runpod/16b_launch_fino_pseudo_pipeline_bg.sh`
- `scripts/runpod/17_status_and_logs.sh`

These wrappers run the Chapter 2 jobs under `nohup`, print a PID, and write logs to `/workspace/logs/*.log`.

## GPU use in Step 04 (DROID conversion)

To better match Chapter 2's **frozen visual backbone** assumption, `convert-droid` can now optionally **precompute frozen visual/context features during conversion**.

- If `ENCODER=fallback`, conversion remains CPU-friendly.
- If `USE_HF=1`, `ENCODER=siglip2_dinov2`, and `EDSVFH_DEVICE=cuda`, Step 04 will use the GPU to precompute per-step frozen features and store them inside the shards.

This means Step 04 can now use the GPU in a way that is consistent with Chapter 2. The raw TFDS / HDF5 I/O part is still storage-bound, so GPU utilization during conversion will not necessarily stay near 100% the whole time.

## Recommended public datasets

1. **DROID** — primary large-scale real-world manipulation pretraining
2. **BridgeData V2** — cross-environment real-robot generalization
3. **LIBERO** — structured benchmark validation
4. **CALVIN** — long-horizon language-conditioned validation
5. **robomimic** — easy-to-adapt public HDF5 format and ablations
6. **FAILURE (FINO)** — public failure supervision
7. **FailGen / AHA-style synthetic failures** — failure augmentation
8. **REASSEMBLE** — contact-rich multimodal validation

## Encoder strategy

### Recommended production encoder

- **Verifier encoder:** `SigLIP2 + DINOv2` fused (`encoder.name=siglip2_dinov2`)
- **Watcher encoder:** `DINOv2` or the same fused encoder at a lower call rate

### Validated local encoder

- **Fallback encoder:** deterministic CPU-only encoder (`encoder.name=fallback`)

The local packaging environment used for this artifact does **not** contain downloaded Hugging Face weights, so the packaged tests validate the end-to-end Chapter 2 pipeline with the fallback encoder. The larger encoder adapters remain available as optional runtime modules.

## What is actually implemented for the large-data route

### DROID

Implemented and locally validated:

- `convert-droid`: read **prepared TFDS / RLDS builder directories** and convert them into `train/`, `calib/`, `eval/` robomimic-compatible shards
- `train-sharded`: incrementally train on those shards without loading the whole dataset into memory
- support for local prepared builder directories and auto-discovery of local builder directories containing `features.json`
- optional support for direct `gs://...` builder directories when your TensorFlow / TFDS environment is configured for GCS access

### FINO / FAILURE

Implemented and locally validated:

- `fit-droid-success-baseline`: fit a success-only, FIPER-inspired normal baseline from DROID success shards
- `label-fino-pseudo-onset`: infer and write `pseudo_failure_onset` for each FINO episode while preserving the original onset
- `convert-fino-manifest`: convert a **FINO-style manifest** (raw or pseudo-onset relabeled) into robomimic-compatible failure shards
- `fine-tune-fino`: load an existing verifier bundle and fine-tune the **failure-horizon heads** on those failure shards
- `rebuild-fino-pseudo-onset`: run baseline fit -> pseudo-onset relabel -> reconvert -> fine-tune in one command
- `convert-mock-failure`: create a validated mock FINO-style manifest dataset for smoke tests

Important: the packaged environment validated the **manifest route** and the **fine-tune script** on a mock failure-rich dataset, not the raw downloaded FINO folder layout itself. The real FINO download is public, but because the offline packaging environment cannot inspect that remote file tree directly, the recommended integration path is:

1. download FINO,
2. create a manifest JSONL that points at your extracted files,
3. optionally run `fit-droid-success-baseline` on DROID success shards,
4. optionally run `label-fino-pseudo-onset` to replace FINO onsets with pseudo-onsets while preserving the original labels,
5. run `convert-fino-manifest`,
6. run `fine-tune-fino`.


## RunPod documentation set

This package now includes the following RunPod-facing documents and helpers:

- `docs/RUNPOD_完整教學.md` — Pod deployment and core Chapter 2 execution path
- `docs/第二章_對齊說明.md` — exact mapping from thesis Chapter 2 sections to code modules and scripts
- `docs/RUNPOD_資料下載_完整指南.md` — full public-data download matrix, including DROID, FINO, BridgeData V2, LIBERO, CALVIN, robomimic, REASSEMBLE, and AHA/FailGen status
- `scripts/runpod/12_verify_project.sh` — low-cost local project verification before you spend GPU hours
- `scripts/runpod/07a_fit_droid_success_baseline.sh` / `07b_label_fino_pseudo_onset.sh` — DROID-success baseline fitting and FINO pseudo-onset relabeling helpers
- `scripts/runpod/07c_run_fino_pseudo_onset_pipeline.sh` — resume-safe FINO pseudo-onset pipeline wrapper (`manifest -> baseline -> relabel -> reconvert -> fine-tune`)

### Verification boundary

What is **fully validated in the packaged environment**:

- all Python unit tests
- the fixture route
- the mock DROID sharded route
- the mock failure / FINO fine-tune route
- manifest generation
- DROID-success baseline fitting and FINO pseudo-onset relabeling
- shell syntax of all RunPod helper scripts

What is **documented but cannot be fully re-verified offline in this packaging environment**:

- real external dataset downloads from DROID / FINO / BridgeData V2 / LIBERO / CALVIN / robomimic servers
- live `gs://...` access against the public DROID bucket from within this offline build container
- Hugging Face weight downloads for `siglip2_dinov2`

## Install

### Core install

```bash
python -m pip install -U pip
python -m pip install -e .
```

### For DROID RLDS / TFDS conversion

```bash
python -m pip install -e ".[tfds]"
```

### For Hugging Face encoders

```bash
python -m pip install -e ".[hf]"
```

### For tests

```bash
python -m pip install -e ".[test]"
```

## Quick validation commands

### Validate the small Chapter 2 reference path

```bash
python -m edsvfh.cli make-fixture --output artifacts/tiny_robomimic_fixture.hdf5
python -m edsvfh.cli train-fixture --fixture artifacts/tiny_robomimic_fixture.hdf5 --output artifacts/public_fixture_bundle.pkl --encoder fallback
python -m edsvfh.cli demo --bundle artifacts/public_fixture_bundle.pkl --dataset artifacts/tiny_robomimic_fixture.hdf5 --episode-index 12
python -m pytest -q
```

### Validate the DROID sharded route with a mock source

```bash
python -m edsvfh.cli convert-mock-droid --output-dir artifacts/mock_droid_shards --num-episodes 20 --episodes-per-shard 4
python -m edsvfh.cli train-sharded --shard-dir artifacts/mock_droid_shards --output artifacts/mock_droid_bundle.pkl --encoder fallback
```

### Validate the v31 DROID not-successful route with a mock raw failure source

```bash
python -m edsvfh.cli convert-mock-failure --root-dir artifacts/mock_failure_manifest --output-dir artifacts/mock_failure_shards --num-episodes 20 --episodes-per-shard 4
python -m edsvfh.cli fit-droid-success-baseline --shard-dir artifacts/mock_droid_shards --output artifacts/mock_droid_success_baseline.pkl --encoder fallback --quantile 0.95
python -m edsvfh.cli label-droid-failure-pseudo-onset --manifest artifacts/mock_failure_manifest/mock_fino_manifest.jsonl --baseline artifacts/mock_droid_success_baseline.pkl --output artifacts/mock_failure_manifest/mock_droid_failure_manifest_pseudo.jsonl --image-size 96
python -m edsvfh.cli convert-droid-failure-manifest --manifest artifacts/mock_failure_manifest/mock_droid_failure_manifest_pseudo.jsonl --output-dir artifacts/mock_droid_failure_pseudo_shards --episodes-per-shard 4 --image-size 96 --prefer-pseudo-onset
python -m edsvfh.cli fine-tune-droid-failure --base-bundle artifacts/mock_droid_bundle.pkl --shard-dir artifacts/mock_droid_failure_pseudo_shards --output artifacts/mock_droid_failure_bundle.pkl --epochs 2
```

## DROID: how to read the dataset in practice

This project supports three practical DROID entry modes.

### Mode A — local prepared TFDS builder directory

Use this when you have already copied the RLDS builder locally and you know the directory that contains `features.json`.

Example:

```bash
python -m edsvfh.cli convert-droid \
  --source data/raw/droid/1.0.1 \
  --output-dir data/converted/droid \
  --episodes-per-shard 64 \
  --image-size 96 \
  --step-stride 2 \
  --action-space raw_action
```

### Mode B — local root that contains multiple builder directories

If your local path is a higher-level root, the converter will recursively search for `features.json` and reconstruct the dataset from all compatible builder directories.

Example:

```bash
python -m edsvfh.cli convert-droid \
  --source data/raw/droid \
  --output-dir data/converted/droid \
  --episodes-per-shard 64 \
  --image-size 96
```

### Mode C — live `gs://...` TFDS prepared builder directory

Use this only when your TensorFlow / TFDS environment is configured for GCS reading.

Example:

```bash
python -m edsvfh.cli convert-droid \
  --source gs://gresearch/robotics/droid/1.0.1 \
  --output-dir data/converted/droid \
  --episodes-per-shard 64 \
  --image-size 96 \
  --step-stride 2
```

If direct GCS reading fails, first verify:

- `tensorflow` and `tensorflow-datasets` are installed,
- your environment can read GCS (for example via `gcloud auth application-default login` or a service account),
- the `--source` path points at the actual prepared builder directory, not only the bucket root.

If you do **not** have a reliable GCS runtime, prefer this workflow instead:

1. use `gsutil` to copy only the builder directory you need to local disk,
2. point `--source` to that local prepared directory,
3. convert to HDF5 shards,
4. optionally delete the original RLDS directory after shard conversion.

### Start from the small DROID debug subset

If storage is tight, begin with the public DROID debug subset:

```bash
gsutil -m cp -r gs://gresearch/robotics/droid_100 data/raw/droid/

python -m edsvfh.cli convert-droid \
  --source data/raw/droid/droid_100 \
  --output-dir data/converted/droid_debug \
  --episodes-per-shard 32 \
  --image-size 96
```

If you download the full `1.0.1` prepared RLDS builder locally, use:

```bash
gsutil -m cp -r gs://gresearch/robotics/droid/1.0.1 data/raw/droid/

python -m edsvfh.cli convert-droid \
  --source data/raw/droid/1.0.1 \
  --output-dir data/converted/droid \
  --episodes-per-shard 64 \
  --image-size 96 \
  --step-stride 2
```

## Train on DROID shards

Once the shards exist, run incremental training:

```bash
python -m edsvfh.cli train-sharded \
  --shard-dir data/converted/droid \
  --output artifacts/droid_bundle.pkl \
  --encoder fallback
```

If storage is extremely constrained, you may consume and delete train shards in a single pass:

```bash
python -m edsvfh.cli train-sharded \
  --shard-dir data/converted/droid \
  --output artifacts/droid_bundle.pkl \
  --encoder fallback \
  --epochs 1 \
  --delete-consumed-train-shards
```


## DROID not-successful: v31 failure-side route

DROID raw data contains episode folders and synchronized recordings. v31 adds a raw-data manifest builder that scans DROID raw paths containing failure markers such as `failure`, `not_successful`, or `unsuccessful`, extracts one RGB stream to PNG frames, converts available low-dimensional HDF5 signals to `.npy`, and writes a failure manifest that can be processed by the existing failure-horizon shard converter.

### Step 1 — generate the DROID not-successful manifest

```bash
python -m edsvfh.cli generate-droid-failure-manifest \
  --root-dir data/raw/droid_raw/1.0.1 \
  --output data/raw/droid_failure/droid_failure_manifest.jsonl \
  --frames-root data/raw/droid_failure/frames \
  --image-size 96 \
  --frame-stride 2
```

### Step 2 — fit the DROID success baseline

```bash
python -m edsvfh.cli fit-droid-success-baseline \
  --shard-dir data/converted/droid_curated \
  --output artifacts/droid_success_baseline.pkl \
  --encoder fallback \
  --feature-source visual \
  --window 3 \
  --phase-bins 10 \
  --quantile 0.97
```

### Step 3 — relabel DROID not-successful episodes with pseudo-onsets

```bash
python -m edsvfh.cli label-droid-failure-pseudo-onset \
  --manifest data/raw/droid_failure/droid_failure_manifest.jsonl \
  --baseline artifacts/droid_success_baseline.pkl \
  --output data/raw/droid_failure/droid_failure_manifest_pseudo_onset.jsonl \
  --image-size 96
```

### Step 4 — convert to failure-horizon shards

```bash
python -m edsvfh.cli convert-droid-failure-manifest \
  --manifest data/raw/droid_failure/droid_failure_manifest_pseudo_onset.jsonl \
  --output-dir data/converted/droid_failure \
  --episodes-per-shard 32 \
  --image-size 96 \
  --prefer-pseudo-onset
```

### Step 5 — fine-tune the failure-horizon heads

```bash
python -m edsvfh.cli fine-tune-droid-failure \
  --base-bundle artifacts/droid_bundle.pkl \
  --shard-dir data/converted/droid_failure \
  --output artifacts/droid_droid_failure_bundle.pkl \
  --epochs 3 \
  --horizons 1,3,5 \
  --update-scaler
```

### One-shot Stage 2 command

```bash
python -m edsvfh.cli rebuild-droid-failure-pseudo-onset \
  --droid-success-shard-dir data/converted/droid_curated \
  --droid-failure-manifest data/raw/droid_failure/droid_failure_manifest.jsonl \
  --base-bundle artifacts/droid_bundle.pkl \
  --baseline-output artifacts/droid_success_baseline.pkl \
  --pseudo-manifest-output data/raw/droid_failure/droid_failure_manifest_pseudo_onset.jsonl \
  --converted-output-dir data/converted/droid_failure \
  --output-bundle artifacts/droid_droid_failure_bundle.pkl \
  --encoder fallback \
  --feature-source visual \
  --window 3 \
  --phase-bins 10 \
  --quantile 0.97 \
  --epochs 3 \
  --horizons 1,3,5
```

### Extended horizons

The larger-horizon script now defaults to the DROID not-successful bundle:

```bash
bash scripts/runpod/22_test_larger_horizons.sh
```

It reuses `LARGER_HORIZONS_BASE_BUNDLE`, which defaults to `artifacts/droid_droid_failure_bundle.pkl`, freezes existing 1/3/5 heads if requested, and trains newly added horizons such as 10/15/30/45/60.

## FINO / FAILURE: legacy fine-tuning route

### Step 1 — download and unpack FINO

Place the extracted data under something like:

```text
data/raw/fino/
├── data/
├── annotations/
└── ...
```

### Step 2 — create a FINO manifest JSONL

Each line describes one episode. The converter accepts either:

- `frame_paths` directly, or
- `frames_dir` + `frame_glob`

Optional `.npy` side channels can be provided for state, actions, object position, goal position, and uncertainty.

Example line:

```json
{
  "episode_id": "fino_ep_0001",
  "split": "train",
  "task": "pick_place_failure",
  "instruction": "Detect whether the manipulation is heading toward failure.",
  "outcome": "failure",
  "failure_onset": 18,
  "frames_dir": "data/raw/fino/episode_0001/rgb",
  "frame_glob": "*.png",
  "eef_npy": "data/raw/fino/episode_0001/eef.npy",
  "gripper_npy": "data/raw/fino/episode_0001/gripper.npy",
  "object_pos_npy": "data/raw/fino/episode_0001/object_pos.npy",
  "goal_pos_npy": "data/raw/fino/episode_0001/goal_pos.npy",
  "action_npy": "data/raw/fino/episode_0001/action.npy",
  "policy_uncertainty_npy": "data/raw/fino/episode_0001/policy_uncertainty.npy"
}
```

Save those rows to:

```text
data/raw/fino/fino_manifest.jsonl
```

### Step 3 — fit a DROID success baseline and relabel FINO with pseudo-onsets

This project now supports a **FIPER-inspired** failure-onset relabeling stage driven by **success-only DROID rollouts**. The implementation uses a success-only normal baseline with two signals — observation anomaly and action / uncertainty anomaly — and flags onset when the two short-window signals become jointly abnormal, following the paper's high-level design. The original FIPER paper detects failures from out-of-distribution observations and elevated action uncertainty calibrated from successful rollouts.

```bash
python -m edsvfh.cli fit-droid-success-baseline \
  --shard-dir data/converted/droid_curated \
  --output artifacts/droid_success_baseline.pkl \
  --encoder fallback \
  --feature-source visual \
  --window 3 \
  --phase-bins 10 \
  --quantile 0.97

python -m edsvfh.cli label-fino-pseudo-onset \
  --manifest data/raw/fino/fino_manifest.jsonl \
  --baseline artifacts/droid_success_baseline.pkl \
  --output data/raw/fino/fino_manifest_pseudo_onset.jsonl \
  --image-size 96
```

Each relabeled manifest row preserves `original_failure_onset`, adds `pseudo_failure_onset`, `pseudo_onset_reason`, and `pseudo_onset_confidence`, and by default replaces `failure_onset` with the pseudo-onset for downstream conversion.

### Step 4 — convert the relabeled FINO manifest into failure shards

```bash
python -m edsvfh.cli convert-fino-manifest \
  --manifest data/raw/fino/fino_manifest_pseudo_onset.jsonl \
  --output-dir data/converted/fino \
  --episodes-per-shard 32 \
  --image-size 96 \
  --prefer-pseudo-onset
```

### Step 5 — fine-tune the failure-horizon heads

Use a DROID-pretrained bundle as the base bundle:

```bash
python -m edsvfh.cli fine-tune-fino \
  --base-bundle artifacts/droid_bundle.pkl \
  --shard-dir data/converted/fino \
  --output artifacts/droid_fino_bundle.pkl \
  --epochs 3
```

If you want the feature scaler to adapt slightly to the failure dataset domain, add:

```bash
--update-scaler
```

### What this script changes

`fine-tune-fino` keeps the Chapter 2 pipeline intact and only updates the **failure-horizon heads** (and their calibrators). The subgoal, completion, done, event-watcher, memory, and decision logic remain unchanged.

This makes the large-data route match the methodology:

1. **pretrain** subgoal / progress / general scene dynamics on DROID,
2. **fit** a success-only normal baseline from DROID success rollouts,
3. **relabel** FINO failures with pseudo-onsets derived from that baseline,
4. **fine-tune** failure-horizon estimation on the pseudo-onset FINO shards,
5. **deploy** the same online event-driven verifier.

### One-shot rebuild command

If you want the entire pseudo-onset route in one CLI call, use:

```bash
python -m edsvfh.cli rebuild-fino-pseudo-onset \
  --droid-shard-dir data/converted/droid_curated \
  --fino-manifest data/raw/fino/fino_manifest.jsonl \
  --base-bundle artifacts/droid_bundle.pkl \
  --baseline-output artifacts/droid_success_baseline.pkl \
  --pseudo-manifest-output data/raw/fino/fino_manifest_pseudo_onset.jsonl \
  --converted-output-dir data/converted/fino \
  --output-bundle artifacts/droid_fino_bundle.pkl \
  --encoder fallback \
  --feature-source visual \
  --window 3 \
  --phase-bins 10 \
  --quantile 0.97 \
  --epochs 3
```

## Suggested full Chapter 2 data route

### Storage-limited path

1. DROID debug / partial RLDS -> `convert-droid`
2. `train-sharded`
3. `generate-droid-failure-manifest` on available raw not-successful episodes
4. `fit-droid-success-baseline` on DROID success shards
5. `label-droid-failure-pseudo-onset` -> `convert-droid-failure-manifest --prefer-pseudo-onset`
6. `fine-tune-droid-failure`
7. evaluate on DROID not-successful held-out split and any external validation episodes

### Larger real-robot path

1. DROID `1.0.1` RLDS -> `convert-droid`
2. DROID raw not-successful trajectories -> `generate-droid-failure-manifest`
3. train base bundle on DROID success/curated shards
4. `fit-droid-success-baseline` on the success shards
5. DROID not-successful -> `label-droid-failure-pseudo-onset` -> `convert-droid-failure-manifest --prefer-pseudo-onset`
6. `fine-tune-droid-failure`
7. deploy with the same API and online-termination runtime

## Project layout

```text
edsvfh_public_v2/
├── README.md
├── artifacts/
├── edsvfh/
│   ├── api.py
│   ├── calibration.py
│   ├── cli.py
│   ├── config.py
│   ├── context.py
│   ├── decision.py
│   ├── droid_convert.py
│   ├── droid_failure.py
│   ├── encoders.py
│   ├── fino_convert.py
│   ├── fino_finetune.py
│   ├── fiper_pseudo_onset.py
│   ├── memory.py
│   ├── models.py
│   ├── pipeline.py
│   ├── pseudo_labels.py
│   ├── public_data.py
│   ├── schemas.py
│   ├── sharded_train.py
│   ├── train_public.py
│   ├── types.py
│   └── watcher.py
└── tests/
```

## What was validated locally before packaging this artifact

The following paths were executed locally:

- tiny robomimic fixture generation and training
- online-termination runtime replay
- mock DROID-style shard conversion
- sharded incremental training
- mock FINO-style manifest generation for legacy compatibility
- DROID raw not-successful manifest generation against a synthetic raw DROID-like episode
- DROID success-baseline fitting for pseudo-onset calibration
- DROID not-successful pseudo-onset relabeling route
- manifest-to-shard conversion
- failure-horizon fine-tuning from a pretrained sharded bundle
- one-shot pseudo-onset rebuild (`rebuild-droid-failure-pseudo-onset`; legacy `rebuild-fino-pseudo-onset` retained)
- API smoke tests
- `python -m pytest -q`

## Important limitations

1. The local packaging container does **not** include the full live DROID raw not-successful release. Real large-scale DROID raw conversion must be run on RunPod or another environment with the downloaded data.
2. The direct `gs://...` DROID path is implemented but depends on your TensorFlow / TFDS / GCS runtime.
3. The DROID not-successful raw route is validated against a synthetic DROID-like raw episode layout and uses the same manifest converter as the legacy FINO route.
4. DROID not-successful trajectories provide weak failure supervision, not dense human frame-level onset labels; the pseudo-onset stage is still required.
5. The pseudo-onset stage is **FIPER-inspired**, not an exact reproduction of the paper's full method stack. It preserves the success-only calibration idea and dual-signal onset trigger, but implements them in a repository-native way over this project's encoder and shard abstractions.
6. The optional `SigLIP2 + DINOv2` adapters are implemented but not executed in the offline packaging environment.


## Google One 2TB Windows operation guide

If you are using the exact Windows paths below:

- Project root: `C:\Users\Sliver\Desktop\FCU\2026paper\edsvfh_public_v2`
- Google Drive root: `G:\我的雲端硬碟`

see `docs/GOOGLE_ONE_2TB_操作指南.md` and the `scripts/windows/*.cmd` helpers.

The Windows helpers are legacy v30-era helpers. For the v31 thesis route, prefer the RunPod DROID not-successful scripts unless you also add equivalent Windows wrappers for `generate-droid-failure-manifest`, `label-droid-failure-pseudo-onset`, `convert-droid-failure-manifest`, and `fine-tune-droid-failure`.


## RunPod notes

- `scripts/runpod/02_convert_droid100.sh` and `scripts/runpod/04_convert_droid_curated.sh` now default to `CUDA_VISIBLE_DEVICES=""` during TFDS conversion. This avoids wasting startup time on TensorFlow PTX JIT when the Pod GPU architecture is newer than the prebuilt TensorFlow wheel.
- If DROID debug training reports `horizon_*_auc = NaN`, that is usually not a CUDA failure. In this project, AUC is set to `NaN` when the evaluation split has fewer than two classes for a horizon target. Use the DROID not-successful fine-tuning route for the v31 paper failure-horizon stage.


### RunPod 中途中斷與壞掉 shard 的續跑

`convert-droid` 現在會在 resume 時自動檢查各 split 的最後一個 shard。若 RunPod 斷線或 Pod 被中止導致最後一個 HDF5 shard 壞掉，系統會自動將其改名為 `*.corrupt.<timestamp>` 並從最後一個可讀 shard 後續跑；不需要手動刪掉整個 `droid_curated` 目錄。若損壞的 shard 不是 trailing shard，程式仍會停止並要求人工檢查，以避免靜默遺失資料。


## RunPod 環境注意事項

- 建議使用 **Runpod PyTorch 2.8.0** 官方模板。
- **不要手動執行** `pip install --upgrade transformers torch torchvision`。這會破壞模板內已驗證的 PyTorch/CUDA 相容組合。
- 若要啟用第二章正式 encoder（`siglip2_dinov2`），請執行 `bash scripts/runpod/00_repair_hf_runtime.sh`；該腳本只會安裝 `transformers/safetensors/huggingface-hub`，**不會升級 torch/torchvision**。
- 若在 Blackwell / RTX PRO 6000 WK 上遇到 cuDNN 初始化問題，預設會啟用 `EDSVFH_DISABLE_CUDNN=1` 以維持 Step 04 的 GPU frozen-feature 預計算穩定性。

## RunPod performance note

Compared with the older v11 conversion flow, the newer Chapter-2-aligned Step 04 can be much slower if it precomputes SigLIP2+DINOv2 features frame-by-frame. This version switches Step 04 to batched frozen-feature extraction on GPU (`EDSVFH_CONVERT_BATCH_SIZE`, default `16`) and keeps cuDNN enabled by default. If the runtime was manually modified, `scripts/runpod/00_repair_hf_runtime.sh` now accepts the known-good RunPod stacks (`torch 2.4 + torchvision 0.19` and `torch 2.8 + torchvision 0.23`) and installs only Hugging Face dependencies. If the torch stack is polluted, run `bash scripts/runpod/00_force_repair_torch_stack.sh cu124` or `bash scripts/runpod/00_force_repair_torch_stack.sh cu128` explicitly before continuing.


> FINO / FAILURE released layout note: the project now supports both `episodes/<episode>/rgb/*.png` and the public `failnet_dataset/rgb_imgs/<task>/<episode-or-images>` layout plus task-specific `*_annotation.txt` files.

## RunPod paper pack helper

For paper writing and discussion, use `bash scripts/runpod/19_pack_paper_results.sh` after the DROID not-successful pseudo-onset pipeline finishes. It collects redacted environment snapshots, key logs, manifest summaries, artifact hashes, and a paper-facing `experiment_summary.json`, then emits `paper_pack_light_*.tar.gz` and `paper_pack_full_*.tar.gz` under `PAPER_PACK_OUTPUT_ROOT`.

Use `bash scripts/runpod/19b_launch_pack_paper_results_bg.sh` for a background run. The helper tails into `PAPER_PACK_LOG` and `scripts/runpod/17_status_and_logs.sh` now includes that log in its snapshot.

## v31 replay and lightweight ablation evaluation

After the DROID not-successful pseudo-onset rerun and larger-horizon bundle have completed, v31 can generate additional Chapter 4 evidence without retraining:

```bash
cd /workspace/edsvfh_public_v31
source scripts/runpod/runpod.env
bash scripts/runpod/23b_launch_eval_replay_protocol_bg.sh
bash scripts/runpod/24b_launch_eval_ablation_suite_bg.sh
```

The replay step writes event-driven vs fixed-rate verifier metrics to `/workspace/logs/replay_protocol_metrics.json` and `.csv`. The ablation step writes raw/calibrated/monotonic horizon-risk metrics to `/workspace/logs/ablation_suite_metrics.json` and `.csv`.

Package everything afterward with:

```bash
bash scripts/runpod/19_pack_paper_results.sh
```
