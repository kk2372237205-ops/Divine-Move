@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"

echo [五子棋标注交接] 正在检查并准备独立标注环境...
echo 不会删除或修改电脑上已有的 Python、OpenCV 或其他项目环境。
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_annotation_env.ps1"
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (
  echo 安装完成。下一步请双击 02_标注环境自检.bat
) else (
  echo 安装未完成，请阅读上方错误信息。
)
pause
exit /b %RESULT%
