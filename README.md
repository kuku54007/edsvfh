# EDSV-FH

EDSV-FH 是一套針對機器人操作資料的事件驅動驗證與失敗時域預測工具。它的核心目標是把大型公開資料集先轉成可分片、可續跑的訓練格式，再建立一個能輸出子目標、完成度、完成判斷，以及多個預測視窗失敗風險的驗證模型。

這個專案的重點不是端到端重訓整個控制器，而是提供一條可重現的資料處理、訓練、弱監督標註、失敗時域微調、離線重播評估與 API 服務流程。

## 主要功能

- 將 DROID prepared RLDS / TFDS 資料轉成 robomimic 相容的 HDF5 分片
- 以成功軌跡建立基礎模型包
- 以成功軌跡擬合正常行為基線
- 從同一份 DROID 資料中擷取 not-successful 軌跡並產生偽起點（pseudo-onset）標註
- 以失敗側分片微調多視窗失敗時域預測頭
- 以暖啟動方式擴充長時域預測視窗
- 執行離線重播評估與輸出變體比較
- 提供 FastAPI 服務與簡易 demo
- 支援 checkpoint / resume 與長時間背景執行腳本

## 專案結構

```text
edsvfh/
  api.py
  cli.py
  droid_convert.py
  droid_failure.py
  eval_protocols.py
  calibration.py
  decision.py
  models.py
  train_public.py
  sharded_train.py
  ...

scripts/runpod/
  00_bootstrap.sh
  04_convert_droid_curated.sh
  05_train_droid_curated.sh
  07c_run_droid_failure_pseudo_onset_pipeline.sh
  22_test_larger_horizons.sh
  23_eval_replay_protocol.sh
  24_eval_ablation_suite.sh
  25_full_paper_path_droid_failure.sh
  30_smoke_test_full_paper_path_droid_failure.sh
  ...

tests/
docs/
pyproject.toml
requirements.txt
README.md
```

## 這個專案不包含什麼

GitHub 上建議只放程式碼、腳本、測試與文件，不要把以下內容直接提交進 repo：

- 原始資料集
- 轉換後的 HDF5 分片
- 模型 bundle / checkpoint
- logs / cache / paper pack
- 私有環境設定檔

建議至少在 `.gitignore` 中排除：

```gitignore
data/
artifacts/
logs/
cache/
checkpoints/
*.hdf5
*.pkl
*.ckpt
scripts/runpod/runpod.env
__pycache__/
.pytest_cache/
```

## 環境需求

- Python 3.10 以上
- Linux 環境優先
- 若要使用 `siglip2_dinov2`，建議使用具 CUDA 的 GPU 環境
- 若只想做本地快速驗證，可使用 `fallback` encoder

## 安裝方式

### 1. 基本安裝

```bash
python -m pip install -U pip
python -m pip install -e .
```

### 2. 安裝 DROID / TFDS 讀取依賴

```bash
python -m pip install -e ".[tfds]"
```

### 3. 安裝 Hugging Face 編碼器依賴

```bash
python -m pip install -e ".[hf]"
```

### 4. 安裝測試依賴

```bash
python -m pip install -e ".[test]"
```

### 5. GPU 建議

若要使用 `siglip2_dinov2`，請先依照你的 CUDA 版本安裝對應的 PyTorch，再安裝本專案：

```bash
# 依你的 CUDA / 平台安裝對應版本
python -m pip install torch torchvision torchaudio
python -m pip install -e ".[hf,tfds,test]"
```

## 快速開始

### A. 最小本地 smoke test

這組指令不需要真實 DROID 資料，可先確認專案是否能正常運作。

```bash
python -m edsvfh.cli make-fixture --output artifacts/tiny_fixture.hdf5
python -m edsvfh.cli train-fixture \
  --fixture artifacts/tiny_fixture.hdf5 \
  --output artifacts/tiny_bundle.pkl \
  --encoder fallback

python -m edsvfh.cli demo \
  --bundle artifacts/tiny_bundle.pkl \
  --dataset artifacts/tiny_fixture.hdf5
```

### B. 啟動 API

```bash
python -m edsvfh.cli serve \
  --bundle artifacts/tiny_bundle.pkl \
  --host 0.0.0.0 \
  --port 8000
```

## 推薦資料流程：DROID success → DROID not-successful

這是目前建議的主流程。它使用同一份 DROID prepared RLDS / TFDS 資料：

- 成功軌跡用來建立基礎模型與正常行為基線
- not-successful 軌跡用來做偽起點標註、失敗時域微調與評估

### 1. 準備 DROID prepared builder 目錄

專案預期的 DROID prepared 資料通常長這樣：

```text
/path/to/droid/1.0.1/
  dataset_info.json
  features.json
  droid_101-train.tfrecord-00000-of-xxxxx
  droid_101-train.tfrecord-00001-of-xxxxx
  ...
```

### 2. 將成功軌跡轉為分片資料

```bash
python -m edsvfh.cli convert-droid \
  --source /path/to/droid/1.0.1 \
  --output-dir data/converted/droid_curated \
  --split train \
  --outcome-filter success \
  --image-size 96 \
  --step-stride 2 \
  --episodes-per-shard 64 \
  --action-space raw_action \
  --precompute-encoder siglip2_dinov2 \
  --precompute-device cuda
```

若只想做 CPU 驗證，可改成：

```bash
python -m edsvfh.cli convert-droid \
  --source /path/to/droid/1.0.1 \
  --output-dir data/converted/droid_curated \
  --split train \
  --outcome-filter success \
  --image-size 96 \
  --step-stride 2 \
  --episodes-per-shard 64 \
  --precompute-encoder fallback
```

### 3. 訓練基礎模型包

```bash
python -m edsvfh.cli train-sharded \
  --shard-dir data/converted/droid_curated \
  --output artifacts/droid_curated_bundle.pkl \
  --encoder siglip2_dinov2 \
  --epochs 1 \
  --horizons 1,3,5
```

### 4. 擬合成功基線

```bash
python -m edsvfh.cli fit-droid-success-baseline \
  --shard-dir data/converted/droid_curated \
  --output artifacts/droid_success_baseline.pkl \
  --encoder siglip2_dinov2 \
  --feature-source visual \
  --window 3 \
  --phase-bins 10 \
  --quantile 0.97 \
  --min-phase-count 8
```

### 5. 從同一份 DROID 資料擷取 not-successful 軌跡

如果你使用的是 prepared RLDS / TFDS 版 DROID：

```bash
python -m edsvfh.cli generate-droid-rlds-failure-manifest \
  --source /path/to/droid/1.0.1 \
  --output data/raw/droid_failure/droid_failure_manifest.jsonl \
  --split train \
  --image-size 96 \
  --frame-stride 2
```

如果你另外有 raw episode 版 DROID，則可使用：

```bash
python -m edsvfh.cli generate-droid-failure-manifest \
  --root-dir /path/to/droid_raw/1.0.1 \
  --output data/raw/droid_failure/droid_failure_manifest.jsonl \
  --frames-root data/raw/droid_failure/frames \
  --image-size 96 \
  --frame-stride 2
```

### 6. 產生偽起點標註

```bash
python -m edsvfh.cli label-droid-failure-pseudo-onset \
  --manifest data/raw/droid_failure/droid_failure_manifest.jsonl \
  --baseline artifacts/droid_success_baseline.pkl \
  --output data/raw/droid_failure/droid_failure_manifest_pseudo_onset.jsonl \
  --image-size 96 \
  --encoder siglip2_dinov2
```

### 7. 將失敗側資料轉為分片

```bash
python -m edsvfh.cli convert-droid-failure-manifest \
  --manifest data/raw/droid_failure/droid_failure_manifest_pseudo_onset.jsonl \
  --output-dir data/converted/droid_failure \
  --episodes-per-shard 32 \
  --image-size 96 \
  --prefer-pseudo-onset
```

### 8. 微調失敗時域模型

```bash
python -m edsvfh.cli fine-tune-droid-failure \
  --base-bundle artifacts/droid_curated_bundle.pkl \
  --shard-dir data/converted/droid_failure \
  --output artifacts/droid_droid_failure_bundle.pkl \
  --encoder siglip2_dinov2 \
  --epochs 3 \
  --horizons 1,3,5 \
  --update-scaler
```

### 9. 長時域擴充

```bash
python -m edsvfh.cli fine-tune-droid-failure \
  --base-bundle artifacts/droid_droid_failure_bundle.pkl \
  --shard-dir data/converted/droid_failure \
  --output artifacts/droid_droid_failure_bundle_horizons_1x3x5x10x15x30x45x60.pkl \
  --encoder siglip2_dinov2 \
  --epochs 3 \
  --horizons 1,3,5,10,15,30,45,60 \
  --freeze-existing-horizons
```

## 離線評估

### 離線重播評估

```bash
python -m edsvfh.eval_protocols replay \
  --bundle artifacts/droid_droid_failure_bundle_horizons_1x3x5x10x15x30x45x60.pkl \
  --shard-dir data/converted/droid_failure \
  --fixed-rates 1,3,5,10,15,30,45,60 \
  --alarm-decisions watch,shield,abstain \
  --output-json logs/replay_protocol_metrics.json \
  --output-csv logs/replay_protocol_metrics.csv
```

### 輸出變體比較

```bash
python -m edsvfh.eval_protocols ablation \
  --shard-dir data/converted/droid_failure \
  --bundle main=artifacts/droid_droid_failure_bundle.pkl \
  --bundle extended=artifacts/droid_droid_failure_bundle_horizons_1x3x5x10x15x30x45x60.pkl \
  --variants calibrated_monotonic,calibrated,raw_monotonic,raw \
  --ece-bins 10 \
  --output-json logs/ablation_suite_metrics.json \
  --output-csv logs/ablation_suite_metrics.csv
```

## RunPod / 背景執行

如果你在雲端 GPU 環境中長時間執行，建議使用 `scripts/runpod/` 內的背景腳本。

### 1. 建立工作目錄

```bash
bash scripts/runpod/00_make_dirs.sh
```

### 2. 複製並調整環境設定

```bash
cp scripts/runpod/runpod.env.example scripts/runpod/runpod.env
source scripts/runpod/runpod.env
```

至少確認以下路徑是否正確：

- `DROID_SOURCE`
- `DROID_CURATED_ROOT`
- `DROID_FAILURE_CONVERTED_ROOT`
- `ARTIFACT_ROOT`
- `LOG_ROOT`

### 3. 先跑完整 smoke test

```bash
bash scripts/runpod/30b_launch_smoke_test_full_paper_path_droid_failure_bg.sh
```

### 4. 監看狀態

```bash
bash scripts/runpod/17_status_and_logs.sh
```

### 5. 正式完整流程

```bash
bash scripts/runpod/25b_launch_full_paper_path_droid_failure_bg.sh
```

這條完整流程會依序處理：

- 成功軌跡轉換
- 基礎模型包建立
- 正常行為基線擬合
- not-successful 軌跡擷取
- 偽起點標註
- 失敗側微調
- 長時域擴充
- 離線重播評估
- 輸出變體比較
- 結果打包

## 常用命令總表

```bash
python -m edsvfh.cli catalog
python -m edsvfh.cli make-fixture
python -m edsvfh.cli train-fixture
python -m edsvfh.cli train-robomimic
python -m edsvfh.cli convert-droid
python -m edsvfh.cli train-sharded
python -m edsvfh.cli fit-droid-success-baseline
python -m edsvfh.cli generate-droid-rlds-failure-manifest
python -m edsvfh.cli generate-droid-failure-manifest
python -m edsvfh.cli label-droid-failure-pseudo-onset
python -m edsvfh.cli convert-droid-failure-manifest
python -m edsvfh.cli fine-tune-droid-failure
python -m edsvfh.cli rebuild-droid-failure-pseudo-onset
python -m edsvfh.cli demo
python -m edsvfh.cli serve
python -m edsvfh.eval_protocols replay
python -m edsvfh.eval_protocols ablation
```

## 注意事項

1. **資料、模型與結果不包含在 repo 中**
   - 這個 repo 只應包含程式碼、文件、測試與範例設定。

2. **Hugging Face 權重建議事先快取**
   - 若使用 `siglip2_dinov2`，建議先登入或事先下載模型，避免遠端環境遇到 API rate limit。

3. **若只想先確認流程可跑，請先用 `fallback` encoder**
   - 本地 CPU 驗證時，`fallback` 較穩定。

4. **長時間流程請使用 checkpoint / resume**
   - CLI 與 RunPod 腳本都支援長時間任務續跑。

5. **離線重播不等於真實閉迴路部署**
   - `replay` 主要用來比較事件驅動與固定頻率驗證策略，不應直接等同於線上控制安全保證。

## License

請依你的公開策略補上實際 License，例如 MIT、Apache-2.0 或 BSD-3-Clause。

## 致謝

若你使用了 DROID 或其他公開資料集，建議在 GitHub repo 中另外補上資料來源與下載說明。
