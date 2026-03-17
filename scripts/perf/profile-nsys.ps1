param(
    [string]$Target = ".\\build\\benchmark.exe",
    [string]$Args = "",
    [string]$OutputDir = "artifacts/profiles"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$reportBase = Join-Path $OutputDir "nsys-$stamp"

Write-Output "Run Nsight Systems with:"
Write-Output ("nsys profile --output " + $reportBase + " " + $Target + " " + $Args)
