# 第二章對齊說明（EDSV-FH ↔ 實作）

本專案以論文第二章 2.4–2.10 為主體，並在 RunPod 場景下加入：

- checkpoint / resume
- ETA / elapsed / throughput
- background-safe launchers
- Step 04 的 frozen visual feature 預計算（可用 GPU）

## 2.4 Event Watcher
- 檔案：`edsvfh/watcher.py`
- 核心對應：
  - `q_t = α1 d_vis + α2 d_stall + α3 d_unc + α4 d_hs`
  - heartbeat verification
- RunPod 對應：
  - Step 04 轉換時可預先算 frozen features
  - Step 05 / Step 08 訓練 watcher-related heads

## 2.5 Event Memory 與 Context Builder
- 檔案：`edsvfh/memory.py`, `edsvfh/context.py`
- 核心對應：
  - 最近事件保留
  - 短期窗口 + 事件記憶 → context vector
- 工程化補充：
  - 若 `convert-droid` 使用 `--precompute-encoder`，則 Step 04 會把 frozen visual/context 相關特徵直接寫進 shard，供後續 context 構建重用

## 2.6 Subgoal State Estimator
- 檔案：`edsvfh/models.py`, `edsvfh/pipeline.py`
- 輸出：
  - `subgoal`
  - `completion`
  - `done`
- 訓練位置：
  - `train-sharded`

## 2.7 Predictive Failure Horizon
- 檔案：`edsvfh/models.py`, `edsvfh/fino_finetune.py`
- 設定：
  - 預設 horizons = `(1, 3, 5)`
  - `fino_finetune` 專門補強 failure-side
- checkpoint：
  - `fine-tune-fino` 每個 shard 預設存一次 checkpoint

## 2.8 Calibration and Decision Layer
- 檔案：`edsvfh/calibration.py`, `edsvfh/decision.py`
- 內容：
  - Platt / constant calibration
  - `continue / watch / abstain / shield`
  - temporal smoothing via `confirm_count`

## 2.9 兩階段訓練

### Stage 1：DROID 預訓練
- `scripts/runpod/04_convert_droid_curated.sh`
- `scripts/runpod/05_train_droid_curated.sh`

本版嚴格對齊第二章的作法是：
1. Step 04 以 **frozen encoder** 把視覺／上下文相關特徵預先寫入 shards
2. Step 05 用 shards 只訓練 watcher / subgoal / completion / done / horizon heads

### Stage 2：FINO failure-horizon fine-tune
- `scripts/runpod/06_generate_fino_manifest.sh`
- `scripts/runpod/07_convert_fino.sh`
- `scripts/runpod/08_finetune_fino.sh`

## 2.10 線上推論與 online termination
- 檔案：`edsvfh/pipeline.py`, `edsvfh/api.py`
- 腳本：
  - `scripts/runpod/09_demo_droid_debug.sh`
  - `scripts/runpod/10_demo_fino.sh`
- 行為：
  - `shield` 或 `abstain` 會終止 autonomy
  - `demo` 預設不使用 `--no-stop`

## checkpoint / resume 與第二章的關係

論文第二章並未把 checkpoint 當核心方法，但對 RunPod 這類租賃式環境，checkpoint 是讓 2.9 的兩階段訓練能穩定完成的必要工程條件。

因此本版加入：
- `convert-droid`：`.convert_state.json`
- `train-sharded`：`*.train_ckpt.pkl`
- `generate-fino-manifest`：`*.manifest_ckpt.json`
- `convert-fino-manifest`：`.convert_fino_state.json`
- `fine-tune-fino`：`*.fino_ckpt.pkl`

重新執行相同腳本時，會預設自動 resume。

## ETA / 成本控制

所有長流程都會把 ETA 印到 `stderr`：
- `convert-droid`
- `train-sharded`
- `generate-fino-manifest`
- `convert-fino-manifest`
- `fine-tune-fino`

在 RunPod 上，建議：
- 用 `scripts/runpod/14_launch_convert_droid_curated_bg.sh`
- 用 `scripts/runpod/15_launch_train_droid_curated_bg.sh`
- 用 `scripts/runpod/16_launch_fino_finetune_bg.sh`
- 用 `scripts/runpod/17_status_and_logs.sh`

這樣即使瀏覽器或 WebTerminal 斷線，也不會中斷第二章主線訓練。


### v17 performance/runtime notes

- Step 04 now uses **batched** frozen-feature extraction on GPU. The batch size is controlled by `EDSVFH_CONVERT_BATCH_SIZE` (default `16`).
- `EDSVFH_DISABLE_CUDNN` now defaults to `0`; the encoder only falls back to disabling cuDNN after a real runtime failure.
- Do **not** manually run `pip install --upgrade torch torchvision`. Use `scripts/runpod/00_repair_hf_runtime.sh` for Hugging Face dependencies only, and use `scripts/runpod/00_force_repair_torch_stack.sh cu124|cu128` only if the torch stack has already been polluted.


### v17 throughput note

The slower-than-v11 behavior came from per-frame SigLIP2+DINOv2 precomputation in Step 04. v17 switches this to **batched GPU precomputation** (`CONVERT_BATCH_SIZE`, default `16`) while keeping TensorFlow off the GPU and cuDNN enabled by default. This preserves the Chapter-2 design while restoring practical conversion speed.
