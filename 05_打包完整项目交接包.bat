@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"

set "HANDOFF_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%HANDOFF_PY%" set "HANDOFF_PY=%~dp0.venv-annotation\Scripts\python.exe"
if not exist "%HANDOFF_PY%" (
  echo 找不到项目 Python 环境。请先双击 01_首次安装标注环境.bat
  pause
  exit /b 1
)

echo 正在生成完整项目交接包，不会修改或删除项目文件……
"%HANDOFF_PY%" -u "%~dp0scripts\package_full_project_handoff.py"
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (
  echo 请把上方生成的 ZIP 原样发给下一位同学，不要覆盖解压到旧项目。
) else (
  echo 打包失败，请阅读上方错误信息。
)
pause
exit /b %RESULT%
