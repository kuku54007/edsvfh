# RunPod v30 replay and ablation evaluation

v30 adds two evaluation-only steps. They do not rerun DROID conversion, DROID baseline fitting, FINO conversion, or FINO fine-tuning.

## Replay protocol

```bash
cd /workspace/edsvfh_public_v30
source scripts/runpod/runpod.env
bash scripts/runpod/23b_launch_eval_replay_protocol_bg.sh
```

Outputs:

```text
/workspace/logs/replay_protocol_eval.log
/workspace/logs/replay_protocol_metrics.json
/workspace/logs/replay_protocol_metrics.csv
/workspace/logs/replay_protocol_metrics_episodes.csv
```

This compares the event-driven verifier against fixed-rate verifier calls at the configured intervals.

## Lightweight ablation suite

```bash
cd /workspace/edsvfh_public_v30
source scripts/runpod/runpod.env
bash scripts/runpod/24b_launch_eval_ablation_suite_bg.sh
```

Outputs:

```text
/workspace/logs/ablation_suite_eval.log
/workspace/logs/ablation_suite_metrics.json
/workspace/logs/ablation_suite_metrics.csv
```

This evaluates calibrated monotonic risk, calibrated risk without monotonic projection, raw monotonic risk, and raw uncalibrated risk.

## Packaging for writing

After both finish:

```bash
bash scripts/runpod/19_pack_paper_results.sh
```

Send the new paper pack plus the replay and ablation JSON/CSV files if they are not already included.
