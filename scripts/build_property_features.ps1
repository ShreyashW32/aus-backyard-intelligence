param(
    [Parameter(Mandatory = $true)][string]$Parcels,
    [Parameter(Mandatory = $true)][string]$Buildings,
    [Parameter(Mandatory = $true)][string]$Dem,
    [string]$Output = "",
    [string]$LgaName = "",
    [string]$IdField = "",
    [string]$LgaField = "",
    [string]$SuburbField = "",
    [string]$AddressField = "",
    [string]$Vegetation = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "Resolve-Python.ps1")
$python = Get-ProjectPython -Root $root

$argsList = @(
    "-m", "aus_backyard_intelligence.pipeline.feature_builder",
    "--parcels", $Parcels,
    "--buildings", $Buildings,
    "--dem", $Dem
)

if ($Output) { $argsList += @("--output", $Output) }
if ($LgaName) { $argsList += @("--lga-name", $LgaName) }
if ($IdField) { $argsList += @("--id-field", $IdField) }
if ($LgaField) { $argsList += @("--lga-field", $LgaField) }
if ($SuburbField) { $argsList += @("--suburb-field", $SuburbField) }
if ($AddressField) { $argsList += @("--address-field", $AddressField) }
if ($Vegetation) { $argsList += @("--vegetation", $Vegetation) }

& $python @argsList
