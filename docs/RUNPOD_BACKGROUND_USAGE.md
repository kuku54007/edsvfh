# RunPod background execution v33 (web terminal safe)

These launchers detach long-running jobs with `setsid + nohup` and store PID / metadata under `${BG_STATE_ROOT}`.
They are intended for RunPod web terminals that may disconnect.

## One-time environment install

```bash
cd /workspace/edsvfh_public_v33
source scripts/runpod/runpod.env
bash scripts/runpod/00_make_dirs.sh
bash scripts/runpod/00b_launch_bootstrap_bg.sh
```

Check install status:

```bash
bash scripts/runpod/17_status_and_logs.sh
tail -f /workspace/logs/bootstrap.log
```

## Full small smoke test in background

```bash
cd /workspace/edsvfh_public_v33
source scripts/runpod/runpod.env
bash scripts/runpod/30b_launch_smoke_test_full_paper_path_droid_failure_bg.sh
```

This smoke test isolates its outputs under `smoke_*` paths and exercises:

- DROID curated conversion
- DROID base training
- DROID failure manifest generation
- pseudo-onset labeling
- failure conversion
- DROID-failure fine-tuning
- larger horizons
- replay evaluation
- ablation evaluation
- paper-pack packaging

## Formal full run in background

```bash
cd /workspace/edsvfh_public_v33
source scripts/runpod/runpod.env

# Formal run: clear all sample limits.
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

`25_full_paper_path_droid_failure.sh` is resume-friendly. It skips completed outputs unless `EDSVFH_FORCE_RERUN=1`.

## Individual background stages

```bash
bash scripts/runpod/14_launch_convert_droid_curated_bg.sh
bash scripts/runpod/15_launch_train_droid_curated_bg.sh
bash scripts/runpod/07c_launch_droid_failure_pseudo_onset_pipeline_bg.sh
bash scripts/runpod/22b_launch_test_larger_horizons_bg.sh
bash scripts/runpod/23b_launch_eval_replay_protocol_bg.sh
bash scripts/runpod/24b_launch_eval_ablation_suite_bg.sh
bash scripts/runpod/19b_launch_pack_paper_results_bg.sh
```

## Status and logs

```bash
bash scripts/runpod/17_status_and_logs.sh
tail -f /workspace/logs/full_paper_path_droid_failure.log
tail -f /workspace/logs/droid_failure_manifest.log
tail -f /workspace/logs/droid_failure_pseudo_onset_pipeline.log
tail -f /workspace/logs/test_droid_failure_larger_horizons_1x3x5x10x15x30x45x60.log
tail -f /workspace/logs/replay_protocol_eval.log
tail -f /workspace/logs/ablation_suite_eval.log
tail -f /workspace/logs/paper_pack_results.log
```

## Important limit variables

- `MAX_EPISODES`: limit for DROID **success** conversion / base training.
- `DROID_FAILURE_SCAN_MAX_EPISODES`: how many RLDS episodes to scan while searching for `/failure/` trajectories.
- `DROID_FAILURE_MAX_EPISODES`: how many failure episodes to keep after scanning.
- `DROID_FAILURE_MAX_FRAMES_PER_EPISODE`: cap per-failure-episode extracted frames.

If you only set `MAX_EPISODES=256`, the success-side DROID conversion is limited, but the failure-side RLDS scan can still traverse the full 95k+ episodes.
