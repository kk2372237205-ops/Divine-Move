$ErrorActionPreference = "Stop"

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$lockFile = Join-Path $projectRoot "requirements-handoff-lock.txt"
$venvRoot = Join-Path $projectRoot ".venv-annotation"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"

Set-Location -LiteralPath $projectRoot

function Test-Python310([string]$Executable, [string[]]$PrefixArguments = @()) {
    if (-not $Executable -or -not (Test-Path -LiteralPath $Executable)) {
        return $null
    }
    try {
        $arguments = @($PrefixArguments) + @(
            "-c",
            "import sys; print(sys.executable if sys.version_info[:2] == (3, 10) else '')"
        )
        $result = (& $Executable @arguments 2>$null | Select-Object -Last 1).Trim()
        if ($LASTEXITCODE -eq 0 -and $result -and (Test-Path -LiteralPath $result)) {
            return [System.IO.Path]::GetFullPath($result)
        }
    }
    catch {
        return $null
    }
    return $null
}

function Find-Python310 {
    $launcher = Get-Command "py.exe" -ErrorAction SilentlyContinue
    if ($launcher) {
        $found = Test-Python310 $launcher.Source @("-3.10")
        if ($found) { return $found }
    }

    foreach ($name in @("python.exe", "python3.exe")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) {
            $found = Test-Python310 $command.Source
            if ($found) { return $found }
        }
    }

    $knownPaths = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe"),
        (Join-Path $env:ProgramFiles "Python310\python.exe")
    )
    if (${env:ProgramFiles(x86)}) {
        $knownPaths += Join-Path ${env:ProgramFiles(x86)} "Python310\python.exe"
    }
    foreach ($path in $knownPaths) {
        $found = Test-Python310 $path
        if ($found) { return $found }
    }
    return $null
}

if (-not (Test-Path -LiteralPath $lockFile)) {
    throw "缺少固定依赖清单：$lockFile"
}

$python = Find-Python310
if (-not $python) {
    Write-Host "没有找到 Python 3.10。其他 Python 版本会保留，不会被卸载或覆盖。" -ForegroundColor Yellow
    $winget = Get-Command "winget.exe" -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "本机也没有 winget，无法自动并行安装 Python 3.10。请从 python.org 安装 64 位 Python 3.10 后重新运行本脚本。"
    }
    Write-Host "正在通过 Windows 软件源并行安装 Python 3.10（仅当前用户）..."
    & $winget.Source install --id Python.Python.3.10 -e --source winget --scope user --accept-package-agreements --accept-source-agreements
    $wingetExitCode = $LASTEXITCODE
    $python = Find-Python310
    if (-not $python) {
        throw "Python 3.10 安装后仍未被找到（winget 退出代码：$wingetExitCode）。请重启电脑后再次运行本脚本。"
    }
}

Write-Host "使用 Python：$python"

if (Test-Path -LiteralPath $venvPython) {
    $existing = Test-Python310 $venvPython
    if (-not $existing) {
        throw "现有 .venv-annotation 不是有效的 Python 3.10 环境。为保护已有内容，本脚本不会删除它；请将该目录改名备份后重试。"
    }
    Write-Host "复用现有项目标注环境：$venvRoot"
}
elseif (Test-Path -LiteralPath $venvRoot) {
    throw "目录 .venv-annotation 已存在但不完整。为保护已有内容，本脚本不会删除或覆盖它；请将该目录改名备份后重试。"
}
else {
    Write-Host "创建独立项目环境：$venvRoot"
    & $python -m venv $venvRoot
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $venvPython)) {
        throw "创建 .venv-annotation 失败。"
    }
}

Write-Host "安装固定版本依赖（只写入 .venv-annotation）..."
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "升级项目环境中的 pip 失败。" }
& $venvPython -m pip install -r $lockFile
if ($LASTEXITCODE -ne 0) { throw "安装项目依赖失败。" }

& $venvPython -c "import cv2, numpy, PIL, serial, tkinter; print('环境导入成功：OpenCV', cv2.__version__)"
if ($LASTEXITCODE -ne 0) { throw "依赖安装后导入检查失败。" }

Write-Host "标注环境已准备完成。系统中的其他 Python/OpenCV 环境没有被修改。" -ForegroundColor Green
