param(
    [int]$Limit = 20
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "Resolve-Python.ps1")
& (Get-ProjectPython -Root $root) -m aus_backyard_intelligence.qgis --list-algorithms --limit $Limit
