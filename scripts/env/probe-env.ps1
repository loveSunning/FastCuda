param(
    [string]$OutputDir = "artifacts/env"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$outFile = Join-Path $OutputDir "env-$stamp.txt"

function Write-Section {
    param(
        [string]$Title,
        [scriptblock]$Block
    )

    Add-Content -Path $outFile -Value ("== " + $Title + " ==")
    try {
        & $Block | Out-String | Add-Content -Path $outFile
    } catch {
        Add-Content -Path $outFile -Value ("ERROR: " + $_.Exception.Message)
    }
    Add-Content -Path $outFile -Value ""
}

function Get-ToolLocation {
    param(
        [string]$Name
    )

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        return ($Name + " not found")
    }

    return $cmd.Source
}

Set-Content -Path $outFile -Value ("timestamp=" + (Get-Date -Format "s"))
Add-Content -Path $outFile -Value ("cwd=" + (Get-Location).Path)
Add-Content -Path $outFile -Value ""

Write-Section -Title "nvidia-smi" -Block { nvidia-smi }
Write-Section -Title "nvcc --version" -Block { nvcc --version }
Write-Section -Title "nvcc path" -Block { Get-ToolLocation nvcc }
Write-Section -Title "ncu path" -Block { Get-ToolLocation ncu }
Write-Section -Title "nsys path" -Block { Get-ToolLocation nsys }
Write-Section -Title "CUDA_PATH env" -Block { Get-ChildItem Env:CUDA_PATH* }

Write-Output $outFile
