$ErrorActionPreference = 'Stop'
$zip = 'C:\Users\23722\Desktop\下棋机械臂资源\下棋机械臂资源-0701.zip'
$out = Join-Path (Get-Location) '_materials_readonly'
if (-not (Test-Path -LiteralPath $out)) { New-Item -ItemType Directory -Path $out | Out-Null }
Expand-Archive -LiteralPath $zip -DestinationPath $out -Force
Write-Output $out
