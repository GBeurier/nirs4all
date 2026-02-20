# scripts/pre-publish.ps1
# Mirrors .github/workflows/pre-publish.yml locally (Windows PowerShell).
#
# Usage:
#   .\scripts\pre-publish.ps1 [OPTIONS]
#
# Options:
#   -SkipTests          Skip the pytest suite
#   -SkipDocs           Skip Sphinx documentation build
#   -SkipExamples       Skip examples runner
#   -SkipBuild          Skip package build / twine check
#   -Only STEP          Run only one step: ruff | mypy | tests | docs | examples | build
#   -ExamplesCat CAT    Example categories to run (default: "user developer reference")
#   -Python PYTHON      Python interpreter to use (default: auto-detect .venv or python)

param(
    [switch]$SkipTests,
    [switch]$SkipDocs,
    [switch]$SkipExamples,
    [switch]$SkipBuild,

    [ValidateSet("", "ruff", "mypy", "tests", "docs", "examples", "build")]
    [string]$Only = "",

    [string]$ExamplesCat = "user developer reference",
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

function Write-Info    { param([string]$Msg) Write-Host "[pre-publish] $Msg" -ForegroundColor Cyan }
function Write-Success { param([string]$Msg) Write-Host "[pre-publish] $Msg" -ForegroundColor Green }
function Write-Warn    { param([string]$Msg) Write-Host "[pre-publish] $Msg" -ForegroundColor Yellow }
function Write-Err     { param([string]$Msg) Write-Host "[pre-publish] $Msg" -ForegroundColor Red }

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 54) -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host ("=" * 54) -ForegroundColor Cyan
}

# ──────────────────────────────────────────────────────────────────────────────
# Resolve repo root (the directory that contains pyproject.toml)
# ──────────────────────────────────────────────────────────────────────────────

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path

if (-not (Test-Path (Join-Path $RepoRoot "pyproject.toml"))) {
    Write-Err "Could not locate pyproject.toml under $RepoRoot"
    exit 1
}

# ──────────────────────────────────────────────────────────────────────────────
# Resolve Python interpreter
# ──────────────────────────────────────────────────────────────────────────────

if ($Python -eq "") {
    $VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path $VenvPython) {
        $Python = $VenvPython
        Write-Info "Using virtual environment Python: $Python"
    } else {
        $Python = "python"
        Write-Info "Using system Python: $Python"
    }
} else {
    Write-Info "Using specified Python: $Python"
}

# ──────────────────────────────────────────────────────────────────────────────
# State tracking
# ──────────────────────────────────────────────────────────────────────────────

$StepResult = @{}   # "pass" | "fail" | "skip"
$StepLog    = @{}   # path to log file

$TmpLogDir = Join-Path ([System.IO.Path]::GetTempPath()) "pre-publish_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Force -Path $TmpLogDir | Out-Null

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Block
    )

    $logFile = Join-Path $TmpLogDir "$Name.log"
    $StepLog[$Name] = $logFile

    Write-Header $Name

    try {
        & $Block 2>&1 | Tee-Object -FilePath $logFile
        if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            throw "Step exited with code $LASTEXITCODE"
        }
        $StepResult[$Name] = "pass"
        Write-Success "$Name - PASSED"
    }
    catch {
        $StepResult[$Name] = "fail"
        $_ | Out-File -FilePath $logFile -Append -Encoding UTF8
        Write-Err "$Name - FAILED  (full log: $logFile)"
    }
}

function Skip-Step {
    param([string]$Name)
    $StepResult[$Name] = "skip"
    Write-Warn "$Name - SKIPPED"
}

function Test-ShouldRun {
    param([string]$Name)
    return ($Only -eq "" -or $Only -eq $Name)
}

# ──────────────────────────────────────────────────────────────────────────────
# Steps
# ──────────────────────────────────────────────────────────────────────────────

Push-Location $RepoRoot

try {

# ── 1. Ruff ──────────────────────────────────────────────────────────────────
if (Test-ShouldRun "ruff") {
    Invoke-Step -Name "ruff" -Block {
        & $Python -m pip install --quiet --upgrade ruff
        if ($LASTEXITCODE -ne 0) { throw "pip install ruff failed" }
        & $Python -m ruff check .
        if ($LASTEXITCODE -ne 0) { throw "ruff check failed" }
    }
} else {
    Skip-Step "ruff"
}

# ── 2. Mypy ──────────────────────────────────────────────────────────────────
if (Test-ShouldRun "mypy") {
    Invoke-Step -Name "mypy" -Block {
        & $Python -m pip install --quiet --upgrade pip setuptools wheel
        & $Python -m pip install --quiet numpy -r requirements-test.txt
        & $Python -m pip install --quiet -e . --no-deps
        & $Python -m pip install --quiet mypy types-PyYAML
        if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
        & $Python -m mypy nirs4all
        if ($LASTEXITCODE -ne 0) { throw "mypy failed" }
    }
} else {
    Skip-Step "mypy"
}

# ── 3. Tests ─────────────────────────────────────────────────────────────────
if (Test-ShouldRun "tests") {
    if ($SkipTests) {
        Skip-Step "tests"
    } else {
        Invoke-Step -Name "tests" -Block {
            & $Python -m pip install --quiet --upgrade pip setuptools wheel
            & $Python -m pip install --quiet numpy -r requirements-test.txt
            & $Python -m pip install --quiet -e . --no-deps
            if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

            Write-Host "--- Serial subset (known write-contention) ---"
            & $Python -m pytest -v `
                tests/integration/pipeline/test_merge_mixed.py `
                tests/integration/pipeline/test_merge_per_branch.py `
                --cov=nirs4all --cov-append
            if ($LASTEXITCODE -ne 0) { throw "serial tests failed" }

            Write-Host "--- Parallel remainder ---"
            & $Python -m pytest -v -n 8 --dist worksteal tests/ `
                --ignore=tests/integration/pipeline/test_merge_mixed.py `
                --ignore=tests/integration/pipeline/test_merge_per_branch.py `
                --cov=nirs4all --cov-append --cov-report=xml
            if ($LASTEXITCODE -ne 0) { throw "parallel tests failed" }
        }
    }
} else {
    Skip-Step "tests"
}

# ── 4. Documentation ─────────────────────────────────────────────────────────
if (Test-ShouldRun "docs") {
    if ($SkipDocs) {
        Skip-Step "docs"
    } else {
        Invoke-Step -Name "docs" -Block {
            & $Python -m pip install --quiet --upgrade pip setuptools wheel
            & $Python -m pip install --quiet -r docs/readthedocs.requirements.txt
            & $Python -m pip install --quiet -e .
            if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

            Push-Location docs
            try {
                & sphinx-build -b html source _build/html --keep-going
                if ($LASTEXITCODE -ne 0) { throw "sphinx-build failed" }
            } finally {
                Pop-Location
            }
        }
    }
} else {
    Skip-Step "docs"
}

# ── 5. Examples ──────────────────────────────────────────────────────────────
if (Test-ShouldRun "examples") {
    if ($SkipExamples) {
        Skip-Step "examples"
    } else {
        Invoke-Step -Name "examples" -Block {
            & $Python -m pip install --quiet --upgrade pip setuptools wheel
            & $Python -m pip install --quiet numpy
            & $Python -m pip install --quiet -r requirements-examples.txt
            & $Python -m pip install --quiet -e .
            if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

            $cats = $ExamplesCat -split '\s+'
            Push-Location examples
            try {
                $runScript = Join-Path $RepoRoot "examples\run_ci_examples.ps1"
                foreach ($cat in $cats) {
                    Write-Host "--- Running category: $cat ---"
                    & $runScript -Category $cat
                    if ($LASTEXITCODE -ne 0) { throw "examples category '$cat' failed" }
                }
            } finally {
                Pop-Location
            }
        }
    }
} else {
    Skip-Step "examples"
}

# ── 6. Package build ─────────────────────────────────────────────────────────
if (Test-ShouldRun "build") {
    if ($SkipBuild) {
        Skip-Step "build"
    } else {
        Invoke-Step -Name "build" -Block {
            & $Python -m pip install --quiet --upgrade build twine
            if ($LASTEXITCODE -ne 0) { throw "pip install build tools failed" }

            & $Python -m build --sdist --wheel --outdir dist
            if ($LASTEXITCODE -ne 0) { throw "build failed" }

            & $Python -m twine check dist/*
            if ($LASTEXITCODE -ne 0) { throw "twine check failed" }

            # Smoke-test the wheel in a temp venv
            $testVenv = Join-Path ([System.IO.Path]::GetTempPath()) "nirs4all_wheel_test"
            try {
                & $Python -m venv $testVenv
                $testPip = Join-Path $testVenv "Scripts\pip.exe"
                $testPython = Join-Path $testVenv "Scripts\python.exe"

                $wheel = Get-ChildItem dist\*.whl | Select-Object -First 1
                & $testPip install --quiet $wheel.FullName
                if ($LASTEXITCODE -ne 0) { throw "wheel install failed" }

                & $testPython -c "import nirs4all; print(f'nirs4all {nirs4all.__version__} installed OK')"
                if ($LASTEXITCODE -ne 0) { throw "wheel smoke test failed" }
            } finally {
                if (Test-Path $testVenv) {
                    Remove-Item -Recurse -Force $testVenv
                }
            }
        }
    }
} else {
    Skip-Step "build"
}

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host ("=" * 63) -ForegroundColor White
Write-Host "           PRE-PUBLISH VALIDATION SUMMARY" -ForegroundColor White
Write-Host ("=" * 63) -ForegroundColor White

$AllPass = $true
$OrderedSteps = @("ruff", "mypy", "tests", "docs", "examples", "build")
$StepLabels = @{
    ruff     = "Ruff          "
    mypy     = "Mypy          "
    tests    = "Tests         "
    docs     = "Documentation "
    examples = "Examples      "
    build    = "Package Build "
}

foreach ($step in $OrderedSteps) {
    $result = if ($StepResult.ContainsKey($step)) { $StepResult[$step] } else { "skip" }
    $label  = $StepLabels[$step]

    switch ($result) {
        "pass" { Write-Host "  $label PASSED"  -ForegroundColor Green }
        "fail" { Write-Host "  $label FAILED"  -ForegroundColor Red;    $AllPass = $false }
        "skip" { Write-Host "  $label SKIPPED" -ForegroundColor Yellow }
    }
}

Write-Host ("=" * 63) -ForegroundColor White

if ($AllPass) {
    Write-Host "  Ready to publish! Create your release now." -ForegroundColor Green
    Write-Host ("=" * 63) -ForegroundColor White
} else {
    Write-Host "  Fix issues above before creating a release." -ForegroundColor Red
    Write-Host ("=" * 63) -ForegroundColor White
    foreach ($step in $OrderedSteps) {
        if ($StepResult.ContainsKey($step) -and $StepResult[$step] -eq "fail") {
            Write-Err "Log for ${step}: $($StepLog[$step])"
        }
    }
}

} finally {
    Pop-Location
    # Clean up temp logs
    if (Test-Path $TmpLogDir) {
        Remove-Item -Recurse -Force $TmpLogDir -ErrorAction SilentlyContinue
    }
}

if (-not $AllPass) { exit 1 }
exit 0
