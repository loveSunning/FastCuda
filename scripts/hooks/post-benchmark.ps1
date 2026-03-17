param(
    [string]$Artifact = ""
)

$ErrorActionPreference = "Stop"

if (-not $Artifact) {
    Write-Output "post-benchmark: no artifact path supplied"
    exit 0
}

if (Test-Path $Artifact) {
    Write-Output ("post-benchmark artifact verified: " + $Artifact)
} else {
    Write-Error ("post-benchmark artifact missing: " + $Artifact)
}
