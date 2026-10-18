param(
    [string]$Module = "pool_3d",
    [string]$Lga = "Moreton Bay",
    [int]$TopK = 3,
    [string]$Output = "",
    [switch]$OpenQGIS
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "Resolve-Python.ps1")
$python = Get-ProjectPython -Root $root

if (-not $Output) {
    $Output = Join-Path $root "artifacts\qgis\ranked_selection.gpkg"
}

$argsList = @(
    "-m", "aus_backyard_intelligence.map_export",
    "--module", $Module,
    "--lga", $Lga,
    "--top-k", $TopK,
    "--output", $Output
)

if ($OpenQGIS) {
    $argsList += "--open-qgis"
}

& $python @argsList
