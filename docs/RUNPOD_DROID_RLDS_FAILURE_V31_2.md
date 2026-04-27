# RunPod DROID RLDS failure pipeline v31.2

Use this when your DROID folder looks like:

```text
/workspace/data/raw/droid/1.0.1/
  dataset_info.json
  features.json
  droid_101-train.tfrecord-00000-...
  droid_101-train.tfrecord-00001-...
```

This is the prepared DROID RLDS/TFDS layout. v31.2 can mine failure trajectories from `episode_metadata.file_path` and `episode_metadata.recording_folderpath`, where DROID keeps the original recording path containing `/success/` or `/failure/`.

## Minimal run

```bash
cd /workspace/edsvfh_public_v31
source scripts/runpod/runpod.env

export DROID_SOURCE=/workspace/data/raw/droid/1.0.1
export DROID_FAILURE_SOURCE_MODE=rlds
export DROID_FAILURE_RLDS_ROOT=/workspace/data/raw/droid/1.0.1

bash scripts/runpod/00_make_dirs.sh
bash scripts/runpod/04_convert_droid_curated.sh
bash scripts/runpod/05_train_droid_curated.sh
bash scripts/runpod/07c_run_droid_failure_pseudo_onset_pipeline.sh
```

## Generate only the DROID not-successful manifest

```bash
cd /workspace/edsvfh_public_v31
source scripts/runpod/runpod.env
export DROID_FAILURE_SOURCE_MODE=rlds
export DROID_FAILURE_RLDS_ROOT=/workspace/data/raw/droid/1.0.1
bash scripts/runpod/06_generate_droid_failure_manifest.sh
head -n 3 /workspace/data/raw/droid_failure/droid_failure_manifest.jsonl
```

## Smoke test with limited episodes

```bash
export MAX_EPISODES=256
export DROID_FAILURE_MAX_EPISODES=64
export DROID_FAILURE_MAX_FRAMES_PER_EPISODE=80
bash scripts/runpod/04_convert_droid_curated.sh
bash scripts/runpod/05_train_droid_curated.sh
bash scripts/runpod/07c_run_droid_failure_pseudo_onset_pipeline.sh
```

## Full run

Clear the limits for the full run:

```bash
export MAX_EPISODES=
export DROID_FAILURE_MAX_EPISODES=
export DROID_FAILURE_MAX_FRAMES_PER_EPISODE=
bash scripts/runpod/13_full_droid_then_droid_failure.sh
```

The full RLDS scan may produce many PNG frames under `/workspace/data/raw/droid_failure/frames`; ensure the pod volume has sufficient free space.
