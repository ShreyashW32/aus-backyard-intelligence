param(
    [string]$Project = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (-not $Project) {
    $Project = Join-Path $root "artifacts\qgis\ranked_selection_3d.qgz"
}

if (-not (Test-Path -LiteralPath $Project)) {
    Write-Error "Project file not found: $Project"
}

$qgisRoot = if ($env:QGIS_ROOT) { $env:QGIS_ROOT } else { Join-Path $env:ProgramFiles "QGIS" }
$qgis = Join-Path $qgisRoot "bin\qgis-ltr.bat"
if (-not (Test-Path -LiteralPath $qgis)) {
    $qgis = Get-ChildItem -Path (Join-Path $env:ProgramFiles "QGIS*") -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName "bin\qgis-ltr.bat" } |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1
}
if (-not $qgis) {
    Write-Error "QGIS launcher not found. Set QGIS_ROOT to your QGIS install directory."
}
$startupCode = Join-Path $root "scripts\qgis_auto_3d.py"

Start-Process -FilePath $qgis -ArgumentList @("--project", $Project, "--code", $startupCode)
