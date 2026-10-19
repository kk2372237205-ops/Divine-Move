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

"%ANNOTATION_PY%" -u "%~dp0scripts\package_annotation_results.py"
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (
  echo 请把上方生成的 ZIP 文件原样发回，不要自行拆分或覆盖文件。
) else (
  echo 打包失败，请阅读上方错误信息。
)
pause
exit /b %RESULT%
