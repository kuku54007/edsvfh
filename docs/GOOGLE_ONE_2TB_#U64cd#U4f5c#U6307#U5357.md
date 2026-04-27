# Google One 2TB 研究資料管理與操作指南（對應第二章）

本指南以以下已知條件為前提：

- 專案根目錄：`C:\Users\Sliver\Desktop\FCU\2026paper\edsvfh_public_v2`
- Google Drive for desktop 掛載位置：`G:\我的雲端硬碟`
- 你已經有可用的 Python 環境與 `data/raw/droid/droid_100`
- 本機沒有 D 槽，因此本機 cache 放在 `C:\Users\Sliver\Desktop\FCU\2026paper\edsvfh_cache`
- 本指南**不再重複**建立 venv 與下載 `droid_100` 的步驟

## 第二章對應關係

這份專案對第二章的對應如下：

1. **DROID 預訓練**
   - Event Watcher
   - Event Memory / Context Builder
   - Subgoal State Estimator
   - Completion / Done
   - 基礎 Failure Horizon
2. **FINO failure fine-tune**
   - Predictive Failure Horizon
   - Calibration + Decision
   - Online termination
3. **Demo / API**
   - 驗證 event-driven online termination 行為

## 0. 先建立雲端與本機資料夾

執行：

```bat
scripts\windows\00_make_google_one_dirs.cmd
```

建立後會有：

- `G:\我的雲端硬碟\EDSVFH-Research\...`
- `C:\Users\Sliver\Desktop\FCU\2026paper\edsvfh_cache\...`

## 1. 用現有的 `droid_100` 產生 DROID debug shards 到 Google One

執行：

```bat
scripts\windows\01_convert_existing_droid100_to_google_one.cmd
```

成功後應出現：

- `G:\我的雲端硬碟\EDSVFH-Research\10_droid_debug\manifest.json`
- `...\train`
- `...\calib`
- `...\eval`

## 2. 從 Google One 複製 shards 到本機 cache，開始第二章的 DROID 預訓練

執行：

```bat
scripts\windows\02_train_droid_debug_from_google_one.cmd
```

這一步已嚴格對應第二章的 DROID success-side 預訓練：

- 事件觸發
- 子目標判斷
- 完成度 / done
- 基礎 horizon
- calibration / decision bundle

## 3. 做 online termination demo

執行：

```bat
scripts\windows\03_demo_droid_debug_online.cmd
```

不要加 `--no-stop`，這樣才符合第二章的 online termination 語義。

## 4. 下載 FINO 到 Google One

請手動到 FINO 官方 GitHub 專案 README 下載：

- Download Data
- Download Annotations

下載後放到：

- `G:\我的雲端硬碟\EDSVFH-Research\30_fino\raw`

建議結構：

```text
G:\我的雲端硬碟\EDSVFH-Research\30_fino\raw\episodes\ep_0001_*\rgb\*.png
G:\我的雲端硬碟\EDSVFH-Research\30_fino\raw\episodes\ep_0001_*\eef.npy
G:\我的雲端硬碟\EDSVFH-Research\30_fino\raw\episodes\ep_0001_*\gripper.npy
G:\我的雲端硬碟\EDSVFH-Research\30_fino\raw\episodes\ep_0001_*\object_pos.npy
G:\我的雲端硬碟\EDSVFH-Research\30_fino\raw\episodes\ep_0001_*\goal_pos.npy
G:\我的雲端硬碟\EDSVFH-Research\30_fino\raw\episodes\ep_0001_*\action.npy
G:\我的雲端硬碟\EDSVFH-Research\30_fino\raw\episodes\ep_0001_*\policy_uncertainty.npy
```

如果 episode 目錄裡有 `metadata.json` / `episode_meta.json` / `labels.json`，manifest 生成器會優先讀取其中的 `failure_onset` 與 `outcome`。若沒有，則會依資料夾名稱是否含 `fail` / `miss` / `drift` / `slip` 推斷是否為 failure。

## 5. 自動產生 FINO manifest

執行：

```bat
scripts\windows\04_generate_fino_manifest.cmd
```

輸出：

- `G:\我的雲端硬碟\EDSVFH-Research\30_fino\manifests\fino_manifest.jsonl`

## 6. 將 FINO 轉成 failure shards

執行：

```bat
scripts\windows\05_convert_fino_to_google_one.cmd
```

輸出：

- `G:\我的雲端硬碟\EDSVFH-Research\30_fino\converted\manifest.json`
- `...\train`
- `...\calib`
- `...\eval`

## 7. 依第二章 failure-horizon 路線做 FINO fine-tune

執行：

```bat
scripts\windows\06_finetune_fino_from_google_one.cmd
```

這一步對應第二章：

- Predictive Failure Horizon
- Calibration + Decision
- Online termination

## 8. 做最終 FINO online demo

執行：

```bat
scripts\windows\07_demo_fino_online.cmd
```

## 9. 每輪做完後清掉本機 cache

執行：

```bat
scripts\windows\08_clean_local_cache.cmd
```

## 10. 正式論文資料擴充（可選）

Google One 2TB 下，不建議保留 full DROID 原始資料副本。建議策略是：

- 以 DROID 公開 bucket 為來源
- 只保留 curated HDF5 shards 在 Google One
- 以新的 curated set 覆蓋舊版 train set

目前程式可用 `--max-episodes` 做 bounded curated subset，例如：

```bat
python -m edsvfh.cli convert-droid ^
  --source gs://gresearch/robotics/droid ^
  --output-dir "G:\我的雲端硬碟\EDSVFH-Research\20_droid_train_curated" ^
  --max-episodes 256 ^
  --episodes-per-shard 64 ^
  --image-size 96 ^
  --step-stride 2 ^
  --action-space raw_action
```

但要注意：這條 live `gs://` 來源路線在程式上有支援，**不是本包內離線驗證的主路徑**。本包已實際驗證的主路徑是：

- 現有本機 `droid_100` → `convert-droid`
- Google One 上的 FINO raw → `generate-fino-manifest` → `convert-fino-manifest`

## 11. 若要更貼近第二章的正式 encoder

目前已實際驗證主線是 `fallback`。如果你要做正式論文主實驗，可安裝 HF 依賴後，把訓練命令中的 `--encoder fallback` 改成：

```bat
--encoder siglip2_dinov2
```

並在 `edsvfh\config.py` 內把 device 設為 `cuda`（若你有 NVIDIA GPU）。
