@echo off
chcp 65001 >nul
setlocal
set "CACHE_ROOT=C:\Users\Sliver\Desktop\FCU\2026paper\edsvfh_cache"
rmdir /s /q "%CACHE_ROOT%\droid_active"
mkdir "%CACHE_ROOT%\droid_active"
rmdir /s /q "%CACHE_ROOT%\fino_active"
mkdir "%CACHE_ROOT%\fino_active"
echo [OK] local cache cleaned.
endlocal
