@echo off
chcp 65001 >nul
setlocal
set "PROJECT_ROOT=C:\Users\Sliver\Desktop\FCU\2026paper\edsvfh_public_v2"
set "GDRIVE_ROOT=G:\我的雲端硬碟\EDSVFH-Research"
cd /d "%PROJECT_ROOT%"
python -m edsvfh.cli label-fino-pseudo-onset ^
  --manifest "%GDRIVE_ROOT%\30_fino\manifests\fino_manifest.jsonl" ^
  --baseline "%GDRIVE_ROOT%\40_artifacts\droid_success_baseline.pkl" ^
  --output "%GDRIVE_ROOT%\30_fino\manifests\fino_manifest_pseudo_onset.jsonl" ^
  --image-size 96
endlocal
