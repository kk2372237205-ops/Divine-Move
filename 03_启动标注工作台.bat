@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"

set "ANNOTATION_PY=%~dp0.venv-annotation\Scripts\python.exe"
if not exist "%ANNOTATION_PY%" set "ANNOTATION_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%ANNOTATION_PY%" (
  echo 找不到项目标注环境。请先双击 01_首次安装标注环境.bat
  pause
  exit /b 1
)

echo 正在启动离线标注工作台。此入口不会打开摄像头、串口或机械臂。
"%ANNOTATION_PY%" -m pc_app.annotation_launcher
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" pause
exit /b %RESULT%
