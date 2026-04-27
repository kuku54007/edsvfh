@echo off
chcp 65001 >nul
setlocal
set "PROJECT_ROOT=C:\Users\Sliver\Desktop\FCU\2026paper\edsvfh_public_v2"
set "GDRIVE_ROOT=G:\我的雲端硬碟\EDSVFH-Research"
cd /d "%PROJECT_ROOT%"
python -m edsvfh.cli fit-droid-success-baseline ^
  --shard-dir "%GDRIVE_ROOT%\20_droid\converted" ^
  --output "%GDRIVE_ROOT%\40_artifacts\droid_success_baseline.pkl" ^
  --encoder fallback ^
  --feature-source visual ^
  --window 3 ^
  --phase-bins 10 ^
  --quantile 0.97 ^
  --min-phase-count 8
endlocal
