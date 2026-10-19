@echo off
setlocal
cd /d "%~dp0"
set "LOG_DIR=logs"
set "LOG_FILE=%LOG_DIR%\run.log"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "PROJECT_PY=.venv\Scripts\python.exe"
if not exist "%PROJECT_PY%" set "PROJECT_PY=.venv-annotation\Scripts\python.exe"
if not exist "%PROJECT_PY%" (
  echo Python environment is missing. Run 01_*.bat first.
  pause
  exit /b 1
)
echo Starting Gomoku Robot desktop app...
echo [%date% %time%] Starting pc_app > "%LOG_FILE%"
"%PROJECT_PY%" -m pc_app >> "%LOG_FILE%" 2>&1
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
  echo.
  echo Startup failed. See %LOG_FILE%
  type "%LOG_FILE%"
  pause
)
exit /b %RESULT%
