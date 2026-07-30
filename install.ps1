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
Write-Host "Next, in order:" -ForegroundColor White
Write-Host ""
Write-Host "  1. Edit your config:" -ForegroundColor White
Write-Host "       $userConfig" -ForegroundColor Gray
Write-Host "     Set audio.input_device to a substring from the device list above." -ForegroundColor Gray
Write-Host ""
Write-Host "  2. In CrewChief: Add/Remove Actions -> bind each intent's action to its key." -ForegroundColor White
Write-Host "     Run this to print the binding sheet:" -ForegroundColor Gray
Write-Host "       .\.venv\Scripts\python.exe -m crew_chief_hearing_aid bindings" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Verify:" -ForegroundColor White
Write-Host "       .\.venv\Scripts\python.exe -m crew_chief_hearing_aid doctor" -ForegroundColor Gray
Write-Host ""
Write-Host "  4. Dry run (logs intents, sends no keys):" -ForegroundColor White
Write-Host "       .\.venv\Scripts\python.exe -m crew_chief_hearing_aid run --dry-run" -ForegroundColor Gray
Write-Host ""
