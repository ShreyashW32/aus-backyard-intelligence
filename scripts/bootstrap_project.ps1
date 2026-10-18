param(
    [switch]$OpenVSCode
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "Resolve-Python.ps1")
$python = Get-ProjectPython -Root $root

$folders = @(
    "artifacts",
    "docs",
    "logs",
    "aus_backyard_intelligence\data\raw",
    "aus_backyard_intelligence\data\interim",
    "aus_backyard_intelligence\data\derived"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Force -Path (Join-Path $root $folder) | Out-Null
}

& $python -m aus_backyard_intelligence.pipeline.sources --write-manifest (Join-Path $root "aus_backyard_intelligence\data\source_manifest.json")

if ($OpenVSCode) {
    & code $root
}

Write-Host "Project bootstrap complete at $root"
