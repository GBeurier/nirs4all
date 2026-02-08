# =============================================================================
# Local CI Examples Runner (Windows PowerShell)
# =============================================================================
# Runs examples with strict output validation and optional parallel workers.
# Uses `ci_example_launcher.py` to enable CI fast-mode caps.
# =============================================================================

param(
    [ValidateSet("user", "developer", "reference", "all")]
    [string]$Category = "all",

    [switch]$Quick,
    [switch]$Strict = $true,
    [switch]$VerboseOutput,
    [switch]$KeepGoing,
    [int]$Jobs = $(if ($env:NIRS4ALL_CI_JOBS) { [int]$env:NIRS4ALL_CI_JOBS } else { 1 })
)

$ErrorActionPreference = "Stop"

if ($Jobs -lt 1) {
    Write-Host "Invalid -Jobs value '$Jobs'. Using 1." -ForegroundColor Yellow
    $Jobs = 1
}

if ($Jobs -gt 1 -and -not $KeepGoing) {
    # Parallel mode cannot cleanly stop on first failure once workers are started.
    $KeepGoing = $true
}

$FastMode = if ($env:NIRS4ALL_EXAMPLE_FAST) { $env:NIRS4ALL_EXAMPLE_FAST } else { "1" }

# =============================================================================
# Error Patterns to Detect
# =============================================================================

$CriticalPatterns = @(
    "Traceback \(most recent call last\)"
    "Error:"
    "ValueError:"
    "TypeError:"
    "KeyError:"
    "ModuleNotFoundError:"
    "ImportError:"
    "AttributeError:"
    "RuntimeError:"
    "AssertionError:"
    "FAILED:"
    "VALIDATION FAILED:"
)

$WarningPatterns = @(
    "MSE: N/A"
    "R²: N/A"
    "RMSE: N/A"
    "Results: N/A"
    "Accuracy: N/A"
    "Accuracy: nan%"
    "R²: nan"
    "Best R²: nan"
    "RMSE = nan"
    "nan,"
    "NaN,"
)

$InvalidResultPatterns = @(
    ": 0 samples"
    ": 0 predictions"
    "No predictions"
    "Empty result"
)

# =============================================================================
# Paths and Runtime Setup
# =============================================================================

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkspaceRoot = Split-Path -Parent $ScriptDir
$OutputDir = Join-Path $ScriptDir "workspace\ci_output"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path $OutputDir "run_$Timestamp"
$StatusDir = Join-Path $RunDir "status"
$WorkspacesDir = Join-Path $RunDir "workspaces"
$SummaryFile = Join-Path $RunDir "summary.txt"
$ErrorsFile = Join-Path $RunDir "errors.txt"

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
New-Item -ItemType Directory -Force -Path $StatusDir | Out-Null
New-Item -ItemType Directory -Force -Path $WorkspacesDir | Out-Null

$VenvPython = Join-Path $WorkspaceRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    Write-Host "Using virtual environment Python: $VenvPython"
    $PythonExe = $VenvPython
}
else {
    Write-Host "Virtual environment not found at $VenvPython; using system python."
    $PythonExe = "python"
}

$Launcher = Join-Path $ScriptDir "ci_example_launcher.py"
if (-not (Test-Path $Launcher)) {
    throw "CI launcher not found at $Launcher"
}

function Write-Log {
    param([string]$Message)
    Write-Host $Message
    Add-Content -Path $SummaryFile -Value $Message
}

function Format-Duration {
    param([int]$TotalSeconds)
    $hours = [int]($TotalSeconds / 3600)
    $minutes = [int](($TotalSeconds % 3600) / 60)
    $seconds = [int]($TotalSeconds % 60)
    return ("{0:D2}:{1:D2}:{2:D2}" -f $hours, $minutes, $seconds)
}

Write-Log "CI Examples Runner - Local Validation"
Write-Log "======================================"
Write-Log "Timestamp: $(Get-Date)"
Write-Log "Category: $Category"
Write-Log "Quick mode: $Quick"
Write-Log "Strict mode: $Strict"
Write-Log "Jobs: $Jobs"
Write-Log "Fast mode: $FastMode"
Write-Log "Output dir: $RunDir"
Write-Log ""

# =============================================================================
# Validation
# =============================================================================

function Test-OutputForErrors {
    param([string]$OutputFile)

    $content = Get-Content -Path $OutputFile -Raw -ErrorAction SilentlyContinue
    if (-not $content) { $content = "" }

    $hasCritical = $false
    $hasWarning = $false
    $hasInvalid = $false
    $issues = @()

    foreach ($pattern in $CriticalPatterns) {
        if ($content -match $pattern) {
            $hasCritical = $true
            $issues += "CRITICAL: Found '$pattern'"
        }
    }

    foreach ($pattern in $WarningPatterns) {
        if ($content -match $pattern) {
            $hasWarning = $true
            $issues += "WARNING: Found '$pattern'"
        }
    }

    foreach ($pattern in $InvalidResultPatterns) {
        if ($content -match $pattern) {
            $hasInvalid = $true
            $issues += "INVALID: Found '$pattern'"
        }
    }

    if ($hasCritical) {
        return @{ Status = 2; Issues = $issues }
    }
    elseif ($hasWarning -or $hasInvalid) {
        return @{ Status = 1; Issues = $issues }
    }
    else {
        return @{ Status = 0; Issues = @() }
    }
}

# =============================================================================
# Example Selection
# =============================================================================

Set-Location $ScriptDir

$UserExamples = @(
    "user/01_getting_started/U01_hello_world.py"
    "user/01_getting_started/U02_basic_regression.py"
    "user/01_getting_started/U03_basic_classification.py"
    "user/01_getting_started/U04_visualization.py"
    "user/02_data_handling/U01_flexible_inputs.py"
    "user/02_data_handling/U02_multi_datasets.py"
    "user/02_data_handling/U03_multi_source.py"
    "user/02_data_handling/U04_wavelength_handling.py"
    "user/02_data_handling/U05_synthetic_data.py"
    "user/02_data_handling/U06_synthetic_advanced.py"
    "user/03_preprocessing/U01_preprocessing_basics.py"
    "user/03_preprocessing/U02_feature_augmentation.py"
    "user/03_preprocessing/U03_sample_augmentation.py"
    "user/03_preprocessing/U04_signal_conversion.py"
    "user/04_models/U01_multi_model.py"
    "user/04_models/U02_hyperparameter_tuning.py"
    "user/04_models/U03_stacking_ensembles.py"
    "user/04_models/U04_pls_variants.py"
    "user/05_cross_validation/U01_cv_strategies.py"
    "user/05_cross_validation/U02_group_splitting.py"
    "user/05_cross_validation/U03_sample_filtering.py"
    "user/05_cross_validation/U04_aggregation.py"
    "user/05_cross_validation/U05_tagging_analysis.py"
    "user/05_cross_validation/U06_exclusion_strategies.py"
    "user/06_deployment/U01_save_load_predict.py"
    "user/06_deployment/U02_export_bundle.py"
    "user/06_deployment/U03_workspace_management.py"
    "user/06_deployment/U04_sklearn_integration.py"
    "user/07_explainability/U01_shap_basics.py"
    "user/07_explainability/U02_shap_sklearn.py"
    "user/07_explainability/U03_feature_selection.py"
)

$DeveloperExamples = @(
    "developer/01_advanced_pipelines/D01_branching_basics.py"
    "developer/01_advanced_pipelines/D02_branching_advanced.py"
    "developer/01_advanced_pipelines/D03_merge_basics.py"
    "developer/01_advanced_pipelines/D04_merge_sources.py"
    "developer/01_advanced_pipelines/D05_meta_stacking.py"
    "developer/02_generators/D01_generator_syntax.py"
    "developer/02_generators/D02_generator_advanced.py"
    "developer/02_generators/D03_generator_iterators.py"
    "developer/02_generators/D04_nested_generators.py"
    "developer/02_generators/D05_synthetic_custom_components.py"
    "developer/02_generators/D06_synthetic_testing.py"
    "developer/02_generators/D07_synthetic_wavenumber_procedural.py"
    "developer/02_generators/D08_synthetic_application_domains.py"
    "developer/02_generators/D09_synthetic_instruments.py"
    "developer/02_generators/D10_synthetic_custom_components.py"
    "developer/02_generators/D11_synthetic_testing.py"
    "developer/02_generators/D12_synthetic_environmental.py"
    "developer/02_generators/D13_synthetic_validation.py"
    "developer/02_generators/D14_synthetic_fitter.py"
    "developer/03_deep_learning/D01_pytorch_models.py"
    "developer/03_deep_learning/D02_jax_models.py"
    "developer/03_deep_learning/D03_tensorflow_models.py"
    "developer/03_deep_learning/D04_framework_comparison.py"
    "developer/04_transfer_learning/D01_transfer_analysis.py"
    "developer/04_transfer_learning/D02_retrain_modes.py"
    "developer/04_transfer_learning/D03_pca_geometry.py"
    "developer/05_advanced_features/D01_metadata_branching.py"
    "developer/05_advanced_features/D02_concat_transform.py"
    "developer/05_advanced_features/D03_repetition_transform.py"
    "developer/06_internals/D01_session_workflow.py"
    "developer/06_internals/D02_custom_controllers.py"
    "developer/01_advanced_pipelines/D06_separation_branches.py"
    "developer/01_advanced_pipelines/D07_value_mapping.py"
    "developer/06_internals/D03_cache_performance.py"
)

$ReferenceExamples = @(
    "reference/R01_pipeline_syntax.py"
    "reference/R02_generator_reference.py"
    "reference/R03_all_keywords.py"
    "reference/R04_legacy_api.py"
)

$DlExamples = @(
    "D01_pytorch_models.py"
    "D02_jax_models.py"
    "D03_tensorflow_models.py"
    "D04_framework_comparison.py"
)

function Test-IsDlExample {
    param([string]$Example)
    $basename = Split-Path -Leaf $Example
    return $DlExamples -contains $basename
}

$SelectedExamples = switch ($Category) {
    "user" { $UserExamples }
    "developer" { $DeveloperExamples }
    "reference" { $ReferenceExamples }
    "all" { $UserExamples + $DeveloperExamples + $ReferenceExamples }
}

$FilteredExamples = @()
foreach ($ex in $SelectedExamples) {
    if (Test-Path $ex) {
        if ($Quick -and (Test-IsDlExample $ex)) {
            continue
        }
        $FilteredExamples += $ex
    }
}
$SelectedExamples = $FilteredExamples

if ($SelectedExamples.Count -eq 0) {
    Write-Log "No examples selected."
    exit 1
}

Write-Log "Examples to run: $($SelectedExamples.Count)"
Write-Log ""

# =============================================================================
# Execution (sequential or parallel)
# =============================================================================

$GlobalStart = Get-Date

function Invoke-ExampleWorker {
    param(
        [int]$Index,
        [string]$Example,
        [string]$PythonExe,
        [string]$Launcher,
        [string]$RunDir,
        [string]$StatusDir,
        [string]$WorkspacesDir,
        [string]$FastMode
    )

    $exampleName = [System.IO.Path]::GetFileNameWithoutExtension($Example)
    $outputFile = Join-Path $RunDir ("{0:D3}_{1}.log" -f $Index, $exampleName)
    $statusFile = Join-Path $StatusDir ("{0:D3}.status" -f $Index)
    $workspaceDir = Join-Path $WorkspacesDir ("{0:D3}_{1}" -f $Index, $exampleName)

    New-Item -ItemType Directory -Force -Path $workspaceDir | Out-Null

    $startTime = Get-Date
    $startIso = $startTime.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $exitCode = 0

    try {
        $env:NIRS4ALL_EXAMPLE_FAST = $FastMode
        $env:NIRS4ALL_WORKSPACE = $workspaceDir
        & $PythonExe $Launcher $Example *> $outputFile
        $exitCode = $LASTEXITCODE
    }
    catch {
        $exitCode = 1
        $_ | Out-File -FilePath $outputFile -Append -Encoding UTF8
    }

    $endTime = Get-Date
    $duration = [math]::Round(($endTime - $startTime).TotalSeconds)
    $endIso = $endTime.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    "{0}|{1}|{2}|{3}|{4}|{5}|{6}" -f $Index, $Example, $outputFile, $exitCode, $duration, $startIso, $endIso |
        Out-File -FilePath $statusFile -Encoding UTF8
    Write-Host ("DONE   [{0}/{1}] {2} :: {3} ({4}s, exit={5})" -f $Index, $SelectedExamples.Count, $endIso, $Example, $duration, $exitCode)
}

if ($Jobs -gt 1) {
    $jobList = @()

    for ($i = 0; $i -lt $SelectedExamples.Count; $i++) {
        while (($jobList | Where-Object { $_.State -eq "Running" }).Count -ge $Jobs) {
            Start-Sleep -Milliseconds 200
        }

        $index = $i + 1
        $example = $SelectedExamples[$i]
        $launchIso = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        Write-Host ("LAUNCH [{0}/{1}] {2} :: {3}" -f $index, $SelectedExamples.Count, $launchIso, $example)

        $job = Start-Job -ScriptBlock {
            param($Index, $Example, $PythonExe, $Launcher, $RunDir, $StatusDir, $WorkspacesDir, $FastMode)
            $ErrorActionPreference = "Stop"

            $exampleName = [System.IO.Path]::GetFileNameWithoutExtension($Example)
            $outputFile = Join-Path $RunDir ("{0:D3}_{1}.log" -f $Index, $exampleName)
            $statusFile = Join-Path $StatusDir ("{0:D3}.status" -f $Index)
            $workspaceDir = Join-Path $WorkspacesDir ("{0:D3}_{1}" -f $Index, $exampleName)

            New-Item -ItemType Directory -Force -Path $workspaceDir | Out-Null

            $startTime = Get-Date
            $startIso = $startTime.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
            $exitCode = 0
            try {
                $env:NIRS4ALL_EXAMPLE_FAST = $FastMode
                $env:NIRS4ALL_WORKSPACE = $workspaceDir
                & $PythonExe $Launcher $Example *> $outputFile
                $exitCode = $LASTEXITCODE
            }
            catch {
                $exitCode = 1
                $_ | Out-File -FilePath $outputFile -Append -Encoding UTF8
            }

            $endTime = Get-Date
            $duration = [math]::Round(($endTime - $startTime).TotalSeconds)
            $endIso = $endTime.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
            "{0}|{1}|{2}|{3}|{4}|{5}|{6}" -f $Index, $Example, $outputFile, $exitCode, $duration, $startIso, $endIso |
                Out-File -FilePath $statusFile -Encoding UTF8
            "DONE   [{0}] {1} :: {2} ({3}s, exit={4})" -f $Index, $endIso, $Example, $duration, $exitCode
        } -ArgumentList $index, $example, $PythonExe, $Launcher, $RunDir, $StatusDir, $WorkspacesDir, $FastMode

        $jobList += $job
    }

    if ($jobList.Count -gt 0) {
        Wait-Job -Job $jobList | Out-Null
        $jobOutput = Receive-Job -Job $jobList
        foreach ($line in $jobOutput) {
            if ($line) { Write-Host $line }
        }
        Remove-Job -Job $jobList | Out-Null
    }
}
else {
    for ($i = 0; $i -lt $SelectedExamples.Count; $i++) {
        $index = $i + 1
        $example = $SelectedExamples[$i]
        $launchIso = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        Write-Host ("LAUNCH [{0}/{1}] {2} :: {3}" -f $index, $SelectedExamples.Count, $launchIso, $example)
        Invoke-ExampleWorker -Index $index -Example $example `
            -PythonExe $PythonExe -Launcher $Launcher `
            -RunDir $RunDir -StatusDir $StatusDir -WorkspacesDir $WorkspacesDir -FastMode $FastMode
    }
}

# =============================================================================
# Validation and Summary
# =============================================================================

$passed = 0
$failed = 0
$warnings = 0
$failedExamples = @()
$warningExamples = @()

for ($i = 0; $i -lt $SelectedExamples.Count; $i++) {
    $index = $i + 1
    $example = $SelectedExamples[$i]
    $statusFile = Join-Path $StatusDir ("{0:D3}.status" -f $index)

    Write-Host -NoNewline "Running: $example ... "

    if (-not (Test-Path $statusFile)) {
        Write-Host "FAILED (missing status file)"
        $failed++
        $failedExamples += "$example (missing status file)"
        continue
    }

    $raw = (Get-Content -Path $statusFile -Raw).Trim()
    $parts = $raw -split '\|', 7

    if ($parts.Count -lt 7) {
        Write-Host "FAILED (invalid status format)"
        $failed++
        $failedExamples += "$example (invalid status format)"
        continue
    }

    $outputFile = $parts[2]
    $exitCode = [int]$parts[3]
    $duration = [int]$parts[4]
    $startIso = $parts[5]
    $endIso = $parts[6]

    if ($exitCode -ne 0) {
        Write-Host "FAILED (exit code: $exitCode, ${duration}s, done: $endIso)"
        $failed++
        $failedExamples += "$example (exit code: $exitCode)"

        Add-Content -Path $ErrorsFile -Value ""
        Add-Content -Path $ErrorsFile -Value "=== $example (exit code: $exitCode) ==="
        if (Test-Path $outputFile) {
            Get-Content -Path $outputFile | Add-Content -Path $ErrorsFile
        }

        if ($VerboseOutput -and (Test-Path $outputFile)) {
            Write-Host "--- Output ---"
            Get-Content -Path $outputFile
            Write-Host "--- End Output ---"
        }

        if (-not $KeepGoing -and $Jobs -eq 1) {
            Write-Host ""
            Write-Host "Stopping on first failure. Use -KeepGoing to continue."
            break
        }
        continue
    }

    $checkResult = Test-OutputForErrors -OutputFile $outputFile

    switch ($checkResult.Status) {
        0 {
            Write-Host "OK (${duration}s, done: $endIso)"
            $passed++
        }
        1 {
            Write-Host "WARNING (${duration}s, done: $endIso)"
            $warnings++
            $warningExamples += $example
            if ($Strict) {
                $failed++
                $failedExamples += "$example (warnings detected)"
            }

            Add-Content -Path $ErrorsFile -Value ""
            Add-Content -Path $ErrorsFile -Value "=== $example (warnings) ==="
            $checkResult.Issues | ForEach-Object { Add-Content -Path $ErrorsFile -Value $_ }
            Add-Content -Path $ErrorsFile -Value "--- Relevant output ---"
            foreach ($pattern in ($WarningPatterns + $InvalidResultPatterns)) {
                Select-String -Path $outputFile -Pattern $pattern -ErrorAction SilentlyContinue |
                    ForEach-Object { Add-Content -Path $ErrorsFile -Value $_.Line }
            }

            if ($VerboseOutput) {
                Write-Host "  Issues:"
                $checkResult.Issues | ForEach-Object { Write-Host "    $_" }
            }

            if ($Strict -and -not $KeepGoing -and $Jobs -eq 1) {
                Write-Host ""
                Write-Host "Stopping on warning (strict mode). Use -KeepGoing to continue."
                break
            }
        }
        2 {
            Write-Host "CRITICAL (${duration}s, done: $endIso)"
            $failed++
            $failedExamples += "$example (critical error)"

            Add-Content -Path $ErrorsFile -Value ""
            Add-Content -Path $ErrorsFile -Value "=== $example (critical) ==="
            if (Test-Path $outputFile) {
                Get-Content -Path $outputFile | Add-Content -Path $ErrorsFile
            }

            if ($VerboseOutput -and (Test-Path $outputFile)) {
                Write-Host "--- Output ---"
                Get-Content -Path $outputFile
                Write-Host "--- End Output ---"
            }

            if (-not $KeepGoing -and $Jobs -eq 1) {
                Write-Host ""
                Write-Host "Stopping on critical error. Use -KeepGoing to continue."
                break
            }
        }
    }
}

$GlobalEnd = Get-Date
$GlobalDuration = [int][math]::Round(($GlobalEnd - $GlobalStart).TotalSeconds)
$GlobalStartIso = $GlobalStart.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$GlobalEndIso = $GlobalEnd.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

Write-Log ""
Write-Log "========================================"
Write-Log "CI VALIDATION SUMMARY"
Write-Log "========================================"
Write-Log "Total examples: $($SelectedExamples.Count)"
Write-Log "Passed: $passed"
Write-Log "Warnings: $warnings"
Write-Log "Failed: $failed"
Write-Log "Started: $GlobalStartIso"
Write-Log "Finished: $GlobalEndIso"
Write-Log ("Elapsed: {0}s ({1})" -f $GlobalDuration, (Format-Duration -TotalSeconds $GlobalDuration))
Write-Log ""

if ($failedExamples.Count -gt 0) {
    Write-Log "FAILED EXAMPLES:"
    foreach ($ex in $failedExamples) {
        Write-Log "  X $ex"
    }
    Write-Log ""
}

if ($warningExamples.Count -gt 0) {
    Write-Log "EXAMPLES WITH WARNINGS:"
    foreach ($ex in $warningExamples) {
        Write-Log "  ! $ex"
    }
    Write-Log ""
}

Write-Log "Detailed logs: $RunDir"
if ((Test-Path $ErrorsFile) -and ((Get-Item $ErrorsFile).Length -gt 0)) {
    Write-Log "Errors file: $ErrorsFile"
}

if ($failed -gt 0) {
    Write-Host ""
    Write-Host "X CI validation FAILED" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "OK CI validation PASSED" -ForegroundColor Green
exit 0
