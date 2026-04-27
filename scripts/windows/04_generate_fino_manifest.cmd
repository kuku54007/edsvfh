@echo off
chcp 65001 >nul
setlocal
set "PROJECT_ROOT=C:\Users\Sliver\Desktop\FCU\2026paper\edsvfh_public_v2"
set "GDRIVE_ROOT=G:\我的雲端硬碟\EDSVFH-Research"
cd /d "%PROJECT_ROOT%"
python -m edsvfh.cli generate-fino-manifest ^
  --root-dir "%GDRIVE_ROOT%\30_fino\raw" ^
  --output "%GDRIVE_ROOT%\30_fino\manifests\fino_manifest.jsonl"
endlocal
