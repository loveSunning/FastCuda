param(
    [string]$Target = ".\\build\\benchmark.exe",
    [string]$Args = "",
    [string]$OutputDir = "artifacts/profiles"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$reportBase = Join-Path $OutputDir "ncu-$stamp"

Write-Output "Run Nsight Compute with:"
Write-Output ("ncu --set full --export " + $reportBase + " " + $Target + " " + $Args)
