@echo off
chcp 65001 >nul
setlocal
set "PROJECT_ROOT=C:\Users\Sliver\Desktop\FCU\2026paper\edsvfh_public_v2"
set "GDRIVE_ROOT=G:\我的雲端硬碟\EDSVFH-Research"
set "CACHE_ROOT=C:\Users\Sliver\Desktop\FCU\2026paper\edsvfh_cache"

mkdir "%GDRIVE_ROOT%" 2>nul
mkdir "%GDRIVE_ROOT%\00_admin" 2>nul
mkdir "%GDRIVE_ROOT%\10_droid_debug" 2>nul
mkdir "%GDRIVE_ROOT%\20_droid_train_curated" 2>nul
mkdir "%GDRIVE_ROOT%\30_fino" 2>nul
mkdir "%GDRIVE_ROOT%\30_fino\raw" 2>nul
mkdir "%GDRIVE_ROOT%\30_fino\manifests" 2>nul
mkdir "%GDRIVE_ROOT%\30_fino\converted" 2>nul
mkdir "%GDRIVE_ROOT%\40_artifacts" 2>nul
mkdir "%GDRIVE_ROOT%\50_logs" 2>nul
mkdir "%GDRIVE_ROOT%\90_archive" 2>nul
mkdir "%CACHE_ROOT%" 2>nul
mkdir "%CACHE_ROOT%\droid_active" 2>nul
mkdir "%CACHE_ROOT%\fino_active" 2>nul
mkdir "%CACHE_ROOT%\temp" 2>nul

echo [OK] directories prepared.
endlocal
