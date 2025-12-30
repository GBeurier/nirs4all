# =============================================================================
# Local CI Examples Runner (Windows PowerShell)
# =============================================================================
# Mimics the GitHub Actions workflow locally for testing examples
# with strict output validation to catch silent errors.
#
# Usage: .\run_ci_examples.ps1 [OPTIONS]
#
# Options:
#   -Category    Category: user, developer, reference, all (default: all)
#   -Quick       Quick mode: skip deep learning examples
#   -Strict      Strict mode: fail on any warning/error pattern (default: true)
#   -Verbose     Verbose: show all output (not just failures)
#   -KeepGoing   Keep going: don't stop on first failure
#
# Examples:
#   .\run_ci_examples.ps1                         # Run all with strict validation
#   .\run_ci_examples.ps1 -Category user          # Run only user examples
#   .\run_ci_examples.ps1 -Category user -Quick   # Quick mode (no DL)
#   .\run_ci_examples.ps1 -Verbose                # Verbose output
# =============================================================================

param(
    [ValidateSet("user", "developer", "reference", "all")]
    [string]$Category = "all",

    [switch]$Quick,
    [switch]$Strict = $true,
    [switch]$VerboseOutput,
    [switch]$KeepGoing
)

$ErrorActionPreference = "Stop"

# =============================================================================
# Error Patterns to Detect
# =============================================================================

# Critical error patterns (always fail)
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

# Warning patterns that indicate potential issues
$WarningPatterns = @(
    "N/A"
    ": nan"
    ": NaN"
    "nan,"
    "NaN,"
    "MSE: N/A"
    "R²: N/A"
    "RMSE: N/A"
    "Results: N/A"
    "Failed to calculate"
    "\[!\]"
    "Warning:"
    "DeprecationWarning:"
    "FutureWarning:"
    "UserWarning:"
)

# Patterns that indicate empty/invalid results
$InvalidResultPatterns = @(
    "0 samples"
    "0 predictions"
    "No predictions"
    "Empty result"
    "length at least 2"
)

# =============================================================================
# Output Directories
# =============================================================================

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkspaceRoot = Split-Path -Parent $ScriptDir
$OutputDir = Join-Path $ScriptDir "workspace\ci_output"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path $OutputDir "run_$Timestamp"

# Activate virtual environment
$VenvDir = Join-Path $WorkspaceRoot ".venv"
$VenvActivate = Join-Path $VenvDir "Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    Write-Host "Activating virtual environment: $VenvDir"
    & $VenvActivate
} else {
    Write-Host "Warning: Virtual environment not found at $VenvDir"
    Write-Host "Using system Python instead."
}

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$SummaryFile = Join-Path $RunDir "summary.txt"
$ErrorsFile = Join-Path $RunDir "errors.txt"

function Write-Log {
    param([string]$Message)
    Write-Host $Message
    Add-Content -Path $SummaryFile -Value $Message
}

Write-Log "CI Examples Runner - Local Validation"
Write-Log "======================================"
Write-Log "Timestamp: $(Get-Date)"
Write-Log "Category: $Category"
Write-Log "Quick mode: $Quick"
Write-Log "Strict mode: $Strict"
Write-Log "Output dir: $RunDir"
Write-Log ""

# =============================================================================
# Check Output for Errors
# =============================================================================

function Test-OutputForErrors {
    param(
        [string]$OutputFile,
        [string]$ExampleName
    )

    $content = Get-Content -Path $OutputFile -Raw -ErrorAction SilentlyContinue
    if (-not $content) { $content = "" }

    $hasCritical = $false
    $hasWarning = $false
    $hasInvalid = $false
    $issues = @()

    # Check critical patterns
    foreach ($pattern in $CriticalPatterns) {
        if ($content -match $pattern) {
            $hasCritical = $true
            $issues += "CRITICAL: Found '$pattern'"
        }
    }

    # Check warning patterns
    foreach ($pattern in $WarningPatterns) {
        if ($content -match $pattern) {
            $hasWarning = $true
            $issues += "WARNING: Found '$pattern'"
        }
    }

    # Check invalid result patterns
    foreach ($pattern in $InvalidResultPatterns) {
        if ($content -match $pattern) {
            $hasInvalid = $true
            $issues += "INVALID: Found '$pattern'"
        }
    }

    # Return status: 0=ok, 1=warning, 2=critical
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
# Build Example List
# =============================================================================

Set-Location $ScriptDir

# User examples
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
)

$ReferenceExamples = @(
    "reference/R01_pipeline_syntax.py"
    "reference/R02_generator_reference.py"
    "reference/R03_all_keywords.py"
    "reference/R04_legacy_api.py"
)

# Deep learning examples to skip in quick mode
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

# Build selected examples list
$SelectedExamples = @()

switch ($Category) {
    "user" { $SelectedExamples = $UserExamples }
    "developer" { $SelectedExamples = $DeveloperExamples }
    "reference" { $SelectedExamples = $ReferenceExamples }
    "all" { $SelectedExamples = $UserExamples + $DeveloperExamples + $ReferenceExamples }
}

# Filter to existing files only
$FilteredExamples = @()
foreach ($ex in $SelectedExamples) {
    if (Test-Path $ex) {
        # Apply quick mode filter
        if ($Quick -and (Test-IsDlExample $ex)) {
            continue
        }
        $FilteredExamples += $ex
    }
}

$SelectedExamples = $FilteredExamples

Write-Log "Examples to run: $($SelectedExamples.Count)"
Write-Log ""

# =============================================================================
# Run Examples with Validation
# =============================================================================

$passed = 0
$failed = 0
$warnings = 0
$failedExamples = @()
$warningExamples = @()

foreach ($example in $SelectedExamples) {
    $exampleName = [System.IO.Path]::GetFileNameWithoutExtension($example)
    $outputFile = Join-Path $RunDir "$exampleName.log"

    Write-Host -NoNewline "Running: $example ... "

    # Run the example and capture output
    $startTime = Get-Date
    $exitCode = 0

    try {
        $output = & python $example 2>&1
        $output | Out-File -FilePath $outputFile -Encoding UTF8
        $exitCode = $LASTEXITCODE
    }
    catch {
        $exitCode = 1
        $_ | Out-File -FilePath $outputFile -Encoding UTF8
    }

    $endTime = Get-Date
    $duration = [math]::Round(($endTime - $startTime).TotalSeconds)

    # Check exit code first
    if ($exitCode -ne 0) {
        Write-Host "FAILED (exit code: $exitCode, ${duration}s)"
        $failed++
        $failedExamples += "$example (exit code: $exitCode)"

        Add-Content -Path $ErrorsFile -Value ""
        Add-Content -Path $ErrorsFile -Value "=== $example (exit code: $exitCode) ==="
        Get-Content -Path $outputFile | Add-Content -Path $ErrorsFile

        if ($VerboseOutput) {
            Write-Host "--- Output ---"
            Get-Content -Path $outputFile
            Write-Host "--- End Output ---"
        }

        if (-not $KeepGoing) {
            Write-Host ""
            Write-Host "Stopping on first failure. Use -KeepGoing to continue."
            break
        }
        continue
    }

    # Check output for error patterns
    $checkResult = Test-OutputForErrors -OutputFile $outputFile -ExampleName $exampleName

    switch ($checkResult.Status) {
        0 {
            Write-Host "OK (${duration}s)"
            $passed++
        }
        1 {
            Write-Host "WARNING (${duration}s)"
            $warnings++
            $warningExamples += $example
            if ($Strict) {
                $failed++
                $failedExamples += "$example (warnings detected)"
            }

            Add-Content -Path $ErrorsFile -Value ""
            Add-Content -Path $ErrorsFile -Value "=== $example (warnings) ==="
            $checkResult.Issues | ForEach-Object { Add-Content -Path $ErrorsFile -Value $_ }

            if ($VerboseOutput) {
                Write-Host "  Issues:"
                $checkResult.Issues | ForEach-Object { Write-Host "    $_" }
            }

            if ($Strict -and -not $KeepGoing) {
                Write-Host ""
                Write-Host "Stopping on warning (strict mode). Use -KeepGoing to continue."
                break
            }
        }
        2 {
            Write-Host "CRITICAL (${duration}s)"
            $failed++
            $failedExamples += "$example (critical error)"

            Add-Content -Path $ErrorsFile -Value ""
            Add-Content -Path $ErrorsFile -Value "=== $example (critical) ==="
            Get-Content -Path $outputFile | Add-Content -Path $ErrorsFile

            if ($VerboseOutput) {
                Write-Host "--- Output ---"
                Get-Content -Path $outputFile
                Write-Host "--- End Output ---"
            }

            if (-not $KeepGoing) {
                Write-Host ""
                Write-Host "Stopping on critical error. Use -KeepGoing to continue."
                break
            }
        }
    }
}

# =============================================================================
# Summary Report
# =============================================================================

Write-Log ""
Write-Log "========================================"
Write-Log "CI VALIDATION SUMMARY"
Write-Log "========================================"
Write-Log "Total examples: $($SelectedExamples.Count)"
Write-Log "Passed: $passed"
Write-Log "Warnings: $warnings"
Write-Log "Failed: $failed"
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

# Exit with appropriate code
if ($failed -gt 0) {
    Write-Host ""
    Write-Host "X CI validation FAILED" -ForegroundColor Red
    exit 1
}
else {
    Write-Host ""
    Write-Host "OK CI validation PASSED" -ForegroundColor Green
    exit 0
}
