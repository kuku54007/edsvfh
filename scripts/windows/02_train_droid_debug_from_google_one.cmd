@echo off
chcp 65001 >nul
setlocal
set "PROJECT_ROOT=C:\Users\Sliver\Desktop\FCU\2026paper\edsvfh_public_v2"
set "GDRIVE_ROOT=G:\我的雲端硬碟\EDSVFH-Research"
set "CACHE_ROOT=C:\Users\Sliver\Desktop\FCU\2026paper\edsvfh_cache"
cd /d "%PROJECT_ROOT%"
robocopy "%GDRIVE_ROOT%\10_droid_debug" "%CACHE_ROOT%\droid_active" /E /MT:8
python -m edsvfh.cli train-sharded ^
  --shard-dir "%CACHE_ROOT%\droid_active" ^
  --output artifacts\droid_debug_bundle.pkl ^
  --encoder fallback
robocopy "artifacts" "%GDRIVE_ROOT%\40_artifacts" /E /MT:8
endlocal
