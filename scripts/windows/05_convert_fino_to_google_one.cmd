@echo off
chcp 65001 >nul
setlocal
set "PROJECT_ROOT=C:\Users\Sliver\Desktop\FCU\2026paper\edsvfh_public_v2"
set "GDRIVE_ROOT=G:\我的雲端硬碟\EDSVFH-Research"
set "MANIFEST_PATH=%GDRIVE_ROOT%\30_fino\manifests\fino_manifest.jsonl"
set "PSEUDO_MANIFEST_PATH=%GDRIVE_ROOT%\30_fino\manifests\fino_manifest_pseudo_onset.jsonl"
set "EXTRA_ARGS="
if exist "%PSEUDO_MANIFEST_PATH%" (
  set "MANIFEST_PATH=%PSEUDO_MANIFEST_PATH%"
  set "EXTRA_ARGS=--prefer-pseudo-onset"
)
cd /d "%PROJECT_ROOT%"
python -m edsvfh.cli convert-fino-manifest ^
  --manifest "%MANIFEST_PATH%" ^
  --output-dir "%GDRIVE_ROOT%\30_fino\converted" ^
  --episodes-per-shard 32 ^
  --image-size 96 ^
  %EXTRA_ARGS%
endlocal
