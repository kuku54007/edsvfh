@echo off
chcp 65001 >nul
setlocal
set "PROJECT_ROOT=C:\Users\Sliver\Desktop\FCU\2026paper\edsvfh_public_v2"
set "CACHE_ROOT=C:\Users\Sliver\Desktop\FCU\2026paper\edsvfh_cache"
cd /d "%PROJECT_ROOT%"
python -m edsvfh.cli demo ^
  --bundle artifacts\droid_fino_bundle.pkl ^
  --dataset "%CACHE_ROOT%\fino_active\eval\fino_eval_0000.hdf5" ^
  --episode-index 0
endlocal
