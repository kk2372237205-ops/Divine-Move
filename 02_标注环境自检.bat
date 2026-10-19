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

echo [1/2] 检查标注数据、依赖和标定（不会连接任何硬件）...
"%ANNOTATION_PY%" -u "%~dp0scripts\check_annotation_handoff.py"
if errorlevel 1 (
  echo.
  echo 自检失败，请不要开始标注，先处理上方问题。
  pause
  exit /b 1
)

echo.
echo [2/2] 运行无硬件自动测试...
"%ANNOTATION_PY%" -m pytest -q
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (
  echo 自检通过。可以双击 03_启动标注工作台.bat
) else (
  echo 自动测试失败，请不要开始标注。
)
pause
exit /b %RESULT%
