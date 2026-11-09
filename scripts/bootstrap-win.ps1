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
$pythonVersion = & $python -c "import platform; print(platform.python_version())"
if ($LASTEXITCODE -or $pythonVersion -ne '3.12.13') {
    throw 'Expected Python 3.12.13 in .venv-win. Move the old environment aside before retrying.'
}
uv pip sync --require-hashes --python $python (Join-Path $root 'requirements-win.lock')
if ($LASTEXITCODE) { throw 'Windows dependency sync failed' }
uv pip install --python $python --no-deps --editable $root
if ($LASTEXITCODE) { throw 'FastH3 package installation failed' }
& (Join-Path $environment 'Scripts/fasth3.exe') doctor
if ($LASTEXITCODE) { throw 'FastH3 environment check failed' }
