# Install the pinned environment from the repository root.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$environment = Join-Path $root '.venv-win'
$python = Join-Path $environment 'Scripts/python.exe'
if (-not $env:UV_CACHE_DIR) { $env:UV_CACHE_DIR = Join-Path $root '.uv-cache' }
if (-not $env:UV_PYTHON_INSTALL_DIR) { $env:UV_PYTHON_INSTALL_DIR = Join-Path $root '.uv-python' }

if (-not (Test-Path -LiteralPath $python)) {
    uv venv --python 3.12.13 $environment
    if ($LASTEXITCODE) { throw 'Python environment creation failed' }
}
# uv supplies the exact Python even when no system Python is installed.
# Version validation and installation are shared with the direct Python entry.
& $python (Join-Path $PSScriptRoot 'bootstrap.py') @args
if ($LASTEXITCODE) { throw 'FastH3 environment setup failed' }
