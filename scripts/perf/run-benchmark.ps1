param(
    [string]$Operator = "gemm",
    [string]$Kernel = "baseline",
    [string]$Shape = "m=1024,n=1024,k=1024",
    [string]$DType = "fp16",
    [int]$Warmup = 20,
    [int]$Iters = 100,
    [string]$OutputDir = "artifacts/benchmarks",
    [string]$BuildDir = "build"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$safeShape = ($Shape -replace "[^a-zA-Z0-9=,_-]", "_")
$outFile = Join-Path $OutputDir "$Operator-$Kernel-$DType-$safeShape-$stamp.json"

$exeCandidates = @(
    (Join-Path $BuildDir "fastcuda_bench.exe"),
    (Join-Path $BuildDir "Release\\fastcuda_bench.exe"),
    (Join-Path $BuildDir "fastcuda_bench")
)

$benchExe = $exeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

$result = [ordered]@{
    timestamp = Get-Date -Format "s"
    operator = $Operator
    kernel = $Kernel
    shape = $Shape
    dtype = $DType
    warmup_iters = $Warmup
    timed_iters = $Iters
    executable = $benchExe
}

if ($benchExe) {
    $output = & $benchExe $Operator $Kernel $Shape $DType
    $result["status"] = "invoked"
    $result["stdout"] = @($output)
} else {
    $result["status"] = "missing_executable"
    $result["notes"] = "Build fastcuda_bench first with CMake before running benchmark wrapper."
}

$result | ConvertTo-Json -Depth 4 | Set-Content -Path $outFile
Write-Output $outFile
