$ErrorActionPreference = "Stop"

Write-Output "Checking profiler tools..."

foreach ($tool in @("ncu", "nsys")) {
    $cmd = Get-Command $tool -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        Write-Output ($tool + " not found")
    } else {
        Write-Output ($tool + " => " + $cmd.Source)
    }
}
