$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# Load KEY=VALUE pairs from .env (git-ignored) without overriding variables already set.
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$' -and $Matches[2].Trim() -ne '') {
            if (-not [Environment]::GetEnvironmentVariable($Matches[1])) {
                [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2].Trim().Trim('"'))
            }
        }
    }
}

. (Join-Path $PSScriptRoot "Resolve-Python.ps1")
& (Get-ProjectPython -Root $root) -m aus_backyard_intelligence @args
