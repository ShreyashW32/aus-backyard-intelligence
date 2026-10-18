function Test-ProjectPython {
    param(
        [Parameter(Mandatory = $true)][string]$Path
    )

    if (-not $Path -or -not (Test-Path -LiteralPath $Path)) {
        return $false
    }

    try {
        & $Path -V *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Set-ProjectPythonPath {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$PythonPath
    )

    $pythonHome = Split-Path -Parent (Split-Path -Parent $PythonPath)
    $sitePackages = Join-Path $pythonHome "Lib\site-packages"

    $entries = @(
        $Root
    )

    if (Test-Path -LiteralPath $sitePackages) {
        $entries += $sitePackages
    }

    if ($env:PYTHONPATH) {
        $entries += $env:PYTHONPATH
    }

    $env:PYTHONPATH = ($entries | Where-Object { $_ } | Select-Object -Unique) -join ";"
}

function Get-ProjectPython {
    param(
        [Parameter(Mandatory = $true)][string]$Root
    )

    $pathPython = $null
    $pathPythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pathPythonCommand) {
        $pathPython = $pathPythonCommand.Source
    }

    $candidates = @(
        $env:QLDR_PYTHON,
        (Join-Path $Root ".venv-codex\Scripts\python.exe"),
        (Join-Path $Root ".venv\Scripts\python.exe"),
        $pathPython,
        "C:\Users\Shreyash\AppData\Local\Programs\Python\Python311\python.exe"
    )

    foreach ($candidate in $candidates) {
        if (-not $candidate) {
            continue
        }
        if (Test-ProjectPython -Path $candidate) {
            Set-ProjectPythonPath -Root $Root -PythonPath $candidate
            return $candidate
        }
    }

    throw "No working Python interpreter was found for this workspace. Set QLDR_PYTHON to a valid Python executable."
}
