$ErrorActionPreference = "Stop"

$envSnapshot = & "scripts/env/probe-env.ps1"
Write-Output ("pre-benchmark env snapshot: " + $envSnapshot)
