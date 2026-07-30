<#
.SYNOPSIS
    One-shot setup for crew_chief_hearing_aid.

.DESCRIPTION
    Idempotent. Safe to re-run. Creates a virtualenv, installs every dependency,
    pre-downloads all models, writes a user config, and runs doctor.

    If PowerShell blocks the script:
        powershell -ExecutionPolicy Bypass -File .\install.ps1

.PARAMETER SkipModels
    Skip model pre-download. Models will fetch lazily on first run instead,
    which moves a ~200MB download into your first race. Not recommended.

.PARAMETER Force
    Recreate the virtualenv from scratch and overwrite the user config.
#>
[CmdletBinding()]
param(
    [switch]$SkipModels,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $root '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }
function Write-Bad($msg)  { Write-Host "    $msg" -ForegroundColor Red }

Write-Host "crew_chief_hearing_aid installer" -ForegroundColor White
Write-Host "-------------------------------"

# --- 1. Python -------------------------------------------------------------
Write-Step "Checking Python"

$python = $null
foreach ($candidate in @('python', 'py -3.12', 'py -3.11', 'py -3')) {
    $parts = $candidate -split ' ', 2
    $exe = (Get-Command $parts[0] -ErrorAction SilentlyContinue)
    if (-not $exe) { continue }
    try {
        $verOut = if ($parts.Count -gt 1) {
            & $parts[0] $parts[1] -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        } else {
            & $parts[0] -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        }
    } catch { continue }
    if (-not $verOut) { continue }
    $parsed = [version]$verOut
    if ($parsed -ge [version]'3.11') {
        $python = $candidate
        Write-Ok "found Python $verOut ($candidate)"
        break
    }
}

if (-not $python) {
    Write-Bad "Python 3.11+ not found."
    Write-Host ""
    Write-Host "    Install it, then re-run this script:" -ForegroundColor Yellow
    Write-Host "        winget install Python.Python.3.12" -ForegroundColor White
    exit 1
}

# --- 2. Virtualenv ---------------------------------------------------------
Write-Step "Setting up virtualenv"

if ($Force -and (Test-Path $venv)) {
    Write-Warn "removing existing .venv (--Force)"
    Remove-Item $venv -Recurse -Force
}

if (Test-Path $venvPython) {
    Write-Ok ".venv already exists"
} else {
    $parts = $python -split ' ', 2
    if ($parts.Count -gt 1) { & $parts[0] $parts[1] -m venv $venv } else { & $parts[0] -m venv $venv }
    if (-not (Test-Path $venvPython)) { Write-Bad "venv creation failed"; exit 1 }
    Write-Ok "created $venv"
}

# --- 3. Dependencies -------------------------------------------------------
Write-Step "Installing dependencies (this takes a few minutes on first run)"

& $venvPython -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { Write-Bad "pip upgrade failed"; exit 1 }

Push-Location $root
try {
    & $venvPython -m pip install -e ".[runtime,dev]"
    if ($LASTEXITCODE -ne 0) { Write-Bad "dependency install failed"; exit 1 }
} finally {
    Pop-Location
}
Write-Ok "dependencies installed"

# --- 4. Models -------------------------------------------------------------
if ($SkipModels) {
    Write-Step "Skipping model download (--SkipModels)"
    Write-Warn "models will download lazily on first run"
} else {
    Write-Step "Pre-downloading models (~200MB, once)"

    # Whisper: pull the configured checkpoint into the faster-whisper cache.
    & $venvPython -c @"
import sys
try:
    from faster_whisper import WhisperModel
    WhisperModel('tiny.en', device='cpu', compute_type='int8')
    print('    whisper tiny.en ready')
except Exception as exc:
    print(f'    ! whisper download failed: {exc}', file=sys.stderr)
    sys.exit(1)
"@
    if ($LASTEXITCODE -ne 0) { Write-Warn "whisper will retry on first run" }

    # openWakeWord ships its feature extractors separately from the wake models.
    & $venvPython -c @"
import sys
try:
    import openwakeword.utils
    openwakeword.utils.download_models()
    print('    openWakeWord models ready')
except Exception as exc:
    print(f'    ! openWakeWord download failed: {exc}', file=sys.stderr)
    sys.exit(1)
"@
    if ($LASTEXITCODE -ne 0) { Write-Warn "openWakeWord will retry on first run" }

    # Silero VAD, fetched and checksummed by our own model manager.
    & $venvPython -c @"
import sys
try:
    from crew_chief_hearing_aid.models import ensure_model
    print(f'    silero VAD ready: {ensure_model(\"silero_vad\")}')
except Exception as exc:
    print(f'    ! silero download failed: {exc}', file=sys.stderr)
    sys.exit(1)
"@
    if ($LASTEXITCODE -ne 0) { Write-Warn "silero VAD will retry on first run" }

    # model2vec static embeddings.
    & $venvPython -c @"
import sys
try:
    from model2vec import StaticModel
    StaticModel.from_pretrained('minishlab/potion-base-8M')
    print('    model2vec embeddings ready')
except Exception as exc:
    print(f'    ! model2vec download failed: {exc}', file=sys.stderr)
    sys.exit(1)
"@
    if ($LASTEXITCODE -ne 0) { Write-Warn "falling back to the offline HashingEmbedder" }
}

# --- 4b. Launcher on PATH --------------------------------------------------
Write-Step "Installing the 'cchear' launcher"

# WindowsApps is on PATH by default on Windows 10/11 and needs no elevation or
# PATH edit, which makes it the least intrusive place for a per-user shim.
$shimDir = Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps'
if (-not (Test-Path $shimDir)) { New-Item -ItemType Directory -Force $shimDir | Out-Null }
$shimPath = Join-Path $shimDir 'cchear.cmd'

# ASCII, no BOM, CRLF: CMD parses batch files in the OEM codepage and emits
# spurious "not recognized as an internal or external command" errors on
# anything else. Learned the hard way.
$shim = @"
@echo off
REM crew_chief_hearing_aid launcher - runs from any directory.
REM Generated by install.ps1. Re-run the installer if you move the repo.
setlocal
set "CCH_REPO=$root"

if exist "%CCH_REPO%\.venv\Scripts\python.exe" (
    set "CCH_PY=%CCH_REPO%\.venv\Scripts\python.exe"
) else (
    set "CCH_PY=python"
)

REM src layout: the package is not importable without this unless pip-installed.
set "PYTHONPATH=%CCH_REPO%\src;%PYTHONPATH%"

"%CCH_PY%" -m crew_chief_hearing_aid %*
exit /b %ERRORLEVEL%
"@

[System.IO.File]::WriteAllText($shimPath, $shim, (New-Object System.Text.ASCIIEncoding))
Write-Ok "installed $shimPath"

if ($env:PATH -split ';' -contains $shimDir) {
    Write-Ok "'cchear' is on PATH — usable from any directory"
} else {
    Write-Warn "$shimDir is not on PATH; add it or call the shim by full path"
}

# --- 5. Self-test ----------------------------------------------------------
Write-Step "Running test suite"
Push-Location $root
try {
    & $venvPython -m pytest -q
    if ($LASTEXITCODE -ne 0) { Write-Bad "tests failed — the install is not healthy"; exit 1 }
} finally {
    Pop-Location
}
Write-Ok "tests passed"

# --- 6. User config --------------------------------------------------------
Write-Step "Writing user config"
$configArgs = @('-m', 'crew_chief_hearing_aid', 'init-config')
if ($Force) { $configArgs += '--force' }
Push-Location $root
try {
    & $venvPython @configArgs
} finally {
    Pop-Location
}

# --- 7. Audio devices ------------------------------------------------------
Write-Step "Audio input devices"
Push-Location $root
try {
    & $venvPython -m crew_chief_hearing_aid --log-level WARNING devices
} finally {
    Pop-Location
}

# --- 8. Doctor -------------------------------------------------------------
Write-Step "Diagnostics"
Push-Location $root
try {
    & $venvPython -m crew_chief_hearing_aid --log-level WARNING doctor
} finally {
    Pop-Location
}

# --- Next steps ------------------------------------------------------------
$userConfig = Join-Path $env:APPDATA 'crew_chief_hearing_aid\config.toml'

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "Run this from anywhere — it walks the whole first-run sequence:" -ForegroundColor White
Write-Host ""
Write-Host "    cchear setup" -ForegroundColor Cyan
Write-Host ""
Write-Host "It picks your microphone, captures a wheel button for push-to-talk," -ForegroundColor Gray
Write-Host "then verifies ONE keypress reaches CrewChief before asking you to bind" -ForegroundColor Gray
Write-Host "all 27 actions — if scancodes never arrive, you find out first." -ForegroundColor Gray
Write-Host ""
Write-Host "Other commands:" -ForegroundColor White
Write-Host "    cchear doctor          check install, bindings, config health" -ForegroundColor Gray
Write-Host "    cchear bindings        print the action -> key sheet" -ForegroundColor Gray
Write-Host "    cchear test-key <id>   fire one keypress" -ForegroundColor Gray
Write-Host "    cchear run --dry-run   full pipeline, logs instead of sending keys" -ForegroundColor Gray
Write-Host "    cchear run             for real" -ForegroundColor Gray
Write-Host ""
