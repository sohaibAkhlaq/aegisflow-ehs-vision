<#
.SYNOPSIS
    One-command development environment setup for AegisFlow EHS (Windows).

.DESCRIPTION
    Verifies the Python toolchain, installs dependencies, installs the package in
    editable mode, creates the local .env, and reports what is still missing.

    By default it reuses the current Python interpreter. Pass -Venv to create an
    isolated .venv instead (slower: torch is ~200 MB, and this machine already has
    every dependency installed globally).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
    powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Venv
#>
[CmdletBinding()]
param(
    [switch]$Venv,
    [switch]$Dev
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    [ok]   $msg" -ForegroundColor Green }
function Write-Warn2($msg){ Write-Host "    [warn] $msg" -ForegroundColor Yellow }

Write-Host "AegisFlow EHS - environment setup" -ForegroundColor Magenta
Write-Host "Repo: $RepoRoot"

# --- 1. Python ------------------------------------------------------------
Write-Step "Checking Python"
$pyVersion = (python --version 2>&1) -replace 'Python\s+', ''
if (-not $pyVersion) { throw "Python not found on PATH. Install Python 3.11 and retry." }
$major, $minor = $pyVersion.Split('.')[0..1]
if ([int]$major -ne 3 -or [int]$minor -lt 11) {
    throw "Python $pyVersion found; this project requires 3.11 or newer."
}
Write-Ok "Python $pyVersion"

# --- 2. Virtual environment (optional) ------------------------------------
$python = 'python'
if ($Venv) {
    Write-Step "Creating virtual environment (.venv)"
    if (-not (Test-Path '.venv')) { python -m venv .venv }
    $python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
    Write-Ok "Using $python"
    Write-Warn2 "Activate it in your shell with: .\.venv\Scripts\Activate.ps1"
} else {
    Write-Step "Using the current interpreter (pass -Venv for an isolated environment)"
}

# --- 3. Dependencies ------------------------------------------------------
Write-Step "Installing dependencies"
$req = if ($Dev) { 'requirements-dev.txt' } else { 'requirements.txt' }
& $python -m pip install --upgrade pip --quiet
& $python -m pip install -r $req
Write-Ok "Installed from $req"

Write-Step "Installing aegisflow in editable mode"
& $python -m pip install -e . --no-deps --quiet
Write-Ok "'aegisflow' CLI available"

# --- 4. Local configuration ----------------------------------------------
Write-Step "Local configuration"
if (-not (Test-Path '.env')) {
    Copy-Item '.env.example' '.env'
    Write-Ok "Created .env from .env.example"
    Write-Warn2 "Add GROQ_API_KEY to .env to enable LLM policy structuring and the VLM tie-break."
    Write-Warn2 "Without a key the pipeline still runs fully: AEGISFLOW_LLM_PROVIDER=offline."
} else {
    Write-Ok ".env already exists (left untouched)"
}

# --- 5. Sanity checks -----------------------------------------------------
Write-Step "Verifying the toolchain"
& $python -c @"
import importlib, sys
mods = ['cv2', 'ultralytics', 'torch', 'fitz', 'fastapi', 'sqlalchemy', 'reportlab', 'groq']
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print('    [fail] missing: ' + ', '.join(missing)); sys.exit(1)
import torch
print(f'    [ok]   torch {torch.__version__}  cuda={torch.cuda.is_available()}')
"@
if ($LASTEXITCODE -ne 0) { throw "Dependency verification failed." }

# --- 6. Dataset -----------------------------------------------------------
Write-Step "Checking the dataset"
if (Test-Path 'data/raw/train') {
    $clips = (Get-ChildItem -Path 'data/raw' -Filter '*.mp4' -Recurse -File).Count
    Write-Ok "$clips clips found under data/raw/"
    if ($clips -lt 691) { Write-Warn2 "Expected 691 clips; the dataset may be incomplete." }
} else {
    Write-Warn2 "data/raw/ is empty."
    Write-Warn2 "Download: https://www.kaggle.com/datasets/trnhhnggiang/videodataset-for-safe-and-unsafe-behaviours"
    Write-Warn2 "Arrange as: data/raw/{train,test}/<class_folder>/*.mp4"
}

# --- 7. Model weights -----------------------------------------------------
Write-Step "Checking model weights"
if (Test-Path 'artifacts/models/yolov8n.pt') {
    Write-Ok "yolov8n.pt present"
} else {
    Write-Warn2 "yolov8n.pt not cached - it downloads automatically (~6 MB) on the first detection run."
}

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host @"

Next steps:
  python -m aegisflow policy parse       Parse the compliance PDF into rules
  python -m aegisflow run --split test   Run the pipeline over the test split
  pytest -m "not slow and not llm"       Fast test loop

Docs: CONTEXT.md (what) | IMPLEMENTATION_PLAN.md (how) | CLAUDE.md (conventions)
"@ -ForegroundColor Gray
