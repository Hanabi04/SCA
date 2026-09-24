@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel% equ 0 (
  py -3 bootstrap.py %*
) else (
  python bootstrap.py %*
)
set "SCA_EXIT=%errorlevel%"
echo.
if not "%SCA_EXIT%"=="0" echo Reproduction failed. Please keep the error message above.
pause
exit /b %SCA_EXIT%
