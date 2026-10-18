param(
    [string]$Package = "",
    [string]$Project = "",
    [switch]$OpenQGIS
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "Resolve-Python.ps1")

if (-not $Package) {
    $Package = Join-Path $root "artifacts\qgis\ranked_selection.gpkg"
}

if (-not $Project) {
    $Project = Join-Path $root "artifacts\qgis\ranked_selection_3d.qgz"
}

$qgisRoot = if ($env:QGIS_ROOT) { $env:QGIS_ROOT } else { Join-Path $env:ProgramFiles "QGIS" }
$qgisPython = Join-Path $qgisRoot "bin\python-qgis-ltr.bat"
if (-not (Test-Path -LiteralPath $qgisPython)) {
    $qgisPython = Get-ChildItem -Path (Join-Path $env:ProgramFiles "QGIS*") -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName "bin\python-qgis-ltr.bat" } |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1
}
if (-not $qgisPython) {
    Write-Error "QGIS Python launcher not found. Set QGIS_ROOT to your QGIS install directory."
}
& $qgisPython (Join-Path $root "scripts\prepare_qgis_3d_project.py") --package $Package --output-project $Project

if ($OpenQGIS) {
    & (Get-ProjectPython -Root $root) -m aus_backyard_intelligence.qgis --open $Project
}
