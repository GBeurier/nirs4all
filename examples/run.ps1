<#
.SYNOPSIS
    nirs4all Examples Runner

.DESCRIPTION
    Run nirs4all examples by category, index, or name pattern.

.PARAMETER Index
    Run single example by index (1-based)

.PARAMETER Begin
    Start from this index (1-based)

.PARAMETER End
    End at this index (1-based)

.PARAMETER Name
    Run by name pattern (glob, e.g., "U01*.py")

.PARAMETER Category
    Category: user, developer, reference, all (default: all)

.PARAMETER Log
    Enable logging to log.txt

.PARAMETER Plot
    Generate plots

.PARAMETER Show
    Show plots interactively

.EXAMPLE
    ./run.ps1                     # Run all examples
    ./run.ps1 -c user             # Run only User path examples
    ./run.ps1 -i 1                # Run first example
    ./run.ps1 -n "U01*.py"        # Run by name pattern
    ./run.ps1 -l -p               # Enable logging and plots
#>

param(
    [Alias('i')]
    [int]$Index = 0,

    [Alias('b')]
    [int]$Begin = 0,

    [Alias('e')]
    [int]$End = 0,

    [Alias('n')]
    [string]$Name = "",

    [Alias('c')]
    [string]$Category = "all",

    [Alias('l')]
    [switch]$Log,

    [Alias('p')]
    [switch]$Plot,

    [Alias('s')]
    [switch]$Show
)

# =============================================================================
# Example Definitions by Category
# =============================================================================

# User path examples (new structure)
$userExamples = @(
    # 01_getting_started
    "user/01_getting_started/U01_hello_world.py"
    "user/01_getting_started/U02_basic_regression.py"
    "user/01_getting_started/U03_basic_classification.py"
    "user/01_getting_started/U04_visualization.py"
    # 02_data_handling
    "user/02_data_handling/U01_flexible_inputs.py"
    "user/02_data_handling/U02_multi_datasets.py"
    "user/02_data_handling/U03_multi_source.py"
    "user/02_data_handling/U04_wavelength_handling.py"
    "user/02_data_handling/U05_synthetic_data.py"
    "user/02_data_handling/U06_synthetic_advanced.py"
    # 03_preprocessing
    "user/03_preprocessing/U01_preprocessing_basics.py"
    "user/03_preprocessing/U02_feature_augmentation.py"
    "user/03_preprocessing/U03_sample_augmentation.py"
    "user/03_preprocessing/U04_signal_conversion.py"
    # 04_models
    "user/04_models/U01_multi_model.py"
    "user/04_models/U02_hyperparameter_tuning.py"
    "user/04_models/U03_stacking_ensembles.py"
    "user/04_models/U04_pls_variants.py"
    # 05_cross_validation
    "user/05_cross_validation/U01_cv_strategies.py"
    "user/05_cross_validation/U02_group_splitting.py"
    "user/05_cross_validation/U03_sample_filtering.py"
    "user/05_cross_validation/U04_aggregation.py"
    # 06_deployment
    "user/06_deployment/U01_save_load_predict.py"
    "user/06_deployment/U02_export_bundle.py"
    "user/06_deployment/U03_workspace_management.py"
    "user/06_deployment/U04_sklearn_integration.py"
    # 07_explainability
    "user/07_explainability/U01_shap_basics.py"
    "user/07_explainability/U02_shap_sklearn.py"
    "user/07_explainability/U03_feature_selection.py"
)

# Developer path examples (new structure)
$developerExamples = @(
    # 01_advanced_pipelines
    "developer/01_advanced_pipelines/D01_branching_basics.py"
    "developer/01_advanced_pipelines/D02_branching_advanced.py"
    "developer/01_advanced_pipelines/D03_merge_basics.py"
    "developer/01_advanced_pipelines/D04_merge_sources.py"
    "developer/01_advanced_pipelines/D05_meta_stacking.py"
    # 02_generators
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
    # 03_deep_learning
    "developer/03_deep_learning/D01_pytorch_models.py"
    "developer/03_deep_learning/D02_jax_models.py"
    "developer/03_deep_learning/D03_tensorflow_models.py"
    "developer/03_deep_learning/D04_framework_comparison.py"
    # 04_transfer_learning
    "developer/04_transfer_learning/D01_transfer_analysis.py"
    "developer/04_transfer_learning/D02_retrain_modes.py"
    "developer/04_transfer_learning/D03_pca_geometry.py"
    # 05_advanced_features
    "developer/05_advanced_features/D01_metadata_branching.py"
    "developer/05_advanced_features/D02_concat_transform.py"
    "developer/05_advanced_features/D03_repetition_transform.py"
    # 06_internals
    "developer/06_internals/D01_session_workflow.py"
    "developer/06_internals/D02_custom_controllers.py"
    "developer/06_internals/D03_cache_performance.py"
)

# Reference examples (new structure)
$referenceExamples = @(
    "reference/R01_pipeline_syntax.py"
    "reference/R02_generator_reference.py"
    "reference/R03_all_keywords.py"
)

# =============================================================================
# Build Examples List
# =============================================================================

# Build selected examples based on category
$examples = @()
switch ($Category.ToLower()) {
    "user" {
        $examples = $userExamples
    }
    "developer" {
        $examples = $developerExamples
    }
    "reference" {
        $examples = $referenceExamples
    }
    "all" {
        $examples = $userExamples + $developerExamples + $referenceExamples
    }
    default {
        Write-Host "Error: Unknown category '$Category'. Valid: user, developer, reference, all" -ForegroundColor Red
        exit 1
    }
}

# Filter to existing files only
$selectedExamples = $examples | Where-Object { Test-Path $_ }

if ($selectedExamples.Count -eq 0) {
    Write-Host "No examples found for category '$Category'." -ForegroundColor Red
    Write-Host "Make sure examples exist in the expected locations."
    exit 1
}

Write-Host "Category: $Category"
Write-Host "Found $($selectedExamples.Count) example(s)"

# =============================================================================
# Setup Logging
# =============================================================================

$logFile = $null
if ($Log) {
    $logFile = Join-Path (Get-Location) "log.txt"
    Write-Host "Logging enabled: $logFile" -ForegroundColor Green
    "=================================================" | Out-File -FilePath $logFile -Encoding UTF8
    "Log started at: $(Get-Date)" | Out-File -FilePath $logFile -Append -Encoding UTF8
    "Category: $Category" | Out-File -FilePath $logFile -Append -Encoding UTF8
    "=================================================" | Out-File -FilePath $logFile -Append -Encoding UTF8
    "" | Out-File -FilePath $logFile -Append -Encoding UTF8
}

# =============================================================================
# Handle Selection Parameters (-i, -b, -e, -n)
# =============================================================================

$paramCount = 0
if ($Index -gt 0) { $paramCount++ }
if ($Begin -gt 0) { $paramCount++ }
if ($End -gt 0) { $paramCount++ }
if ($Name -ne "") { $paramCount++ }

if ($paramCount -gt 1) {
    Write-Host "Error: Specify only one of -i (Index), -b (Begin), -e (End), or -n (Name)." -ForegroundColor Red
    exit 1
}

if ($Index -gt 0) {
    if ($Index -lt 1 -or $Index -gt $selectedExamples.Count) {
        Write-Host "Error: Index $Index is out of range. Valid range is 1..$($selectedExamples.Count)." -ForegroundColor Red
        exit 1
    }
    $selectedExamples = @($selectedExamples[$Index - 1])
    Write-Host "Running single example #$Index: $($selectedExamples[0])"
}
elseif ($Begin -gt 0) {
    if ($Begin -lt 1 -or $Begin -gt $selectedExamples.Count) {
        Write-Host "Error: Start index $Begin is out of range. Valid range is 1..$($selectedExamples.Count)." -ForegroundColor Red
        exit 1
    }
    $startIndex = $Begin - 1
    if ($End -gt 0) {
        if ($End -lt $Begin -or $End -gt $selectedExamples.Count) {
            Write-Host "Error: End index $End is invalid. Valid range is $Begin..$($selectedExamples.Count)." -ForegroundColor Red
            exit 1
        }
        $endIndex = $End - 1
        $selectedExamples = $selectedExamples[$startIndex..$endIndex]
    }
    else {
        $selectedExamples = $selectedExamples[$startIndex..($selectedExamples.Count - 1)]
    }
    Write-Host "Running examples from #$Begin (count: $($selectedExamples.Count))"
}
elseif ($Name -ne "") {
    $matched = $selectedExamples | Where-Object {
        $basename = Split-Path -Leaf $_
        ($basename -like $Name) -or ($_ -like "*$Name*")
    }
    if (-not $matched -or $matched.Count -eq 0) {
        Write-Host "Error: No example matching '$Name' found." -ForegroundColor Red
        Write-Host "Available examples:" -ForegroundColor Yellow
        $selectedExamples | ForEach-Object { Write-Host "  $_" }
        exit 1
    }
    $selectedExamples = $matched
    Write-Host "Running $($selectedExamples.Count) example(s) matching: $Name"
}

# =============================================================================
# Run Examples
# =============================================================================

# Disable emojis when logging to avoid encoding issues
if ($Log) {
    $env:DISABLE_EMOJI = "1"
    Write-Host "Emojis disabled for logging" -ForegroundColor Yellow
}
else {
    Remove-Item Env:\DISABLE_EMOJI -ErrorAction SilentlyContinue
}

# Track results
$failedExamples = @()
$passedExamples = @()

foreach ($example in $selectedExamples) {
    if (Test-Path $example) {
        $startTime = Get-Date
        Write-Host ""
        Write-Host "########################################"
        Write-Host "Launch: $example"
        Write-Host "########################################"

        $args = @()
        if ($Plot) { $args += "--plots" }
        if ($Show) { $args += "--show" }

        $exitCode = 0
        try {
            if ($logFile) {
                "===============================================" | Out-File -FilePath $logFile -Append -Encoding UTF8
                "Starting: $example at $startTime" | Out-File -FilePath $logFile -Append -Encoding UTF8
                "===============================================" | Out-File -FilePath $logFile -Append -Encoding UTF8

                & python $example @args 2>&1 | Tee-Object -FilePath $logFile -Append
                $exitCode = $LASTEXITCODE

                "Finished: $example at $(Get-Date) (exit code: $exitCode)" | Out-File -FilePath $logFile -Append -Encoding UTF8
                "" | Out-File -FilePath $logFile -Append -Encoding UTF8
            }
            else {
                & python $example @args
                $exitCode = $LASTEXITCODE
            }
        }
        catch {
            $exitCode = 1
            Write-Host "Exception: $_" -ForegroundColor Red
        }

        $endTime = Get-Date
        $duration = $endTime - $startTime

        if ($exitCode -ne 0) {
            $failedExamples += "$example (exit code: $exitCode)"
            Write-Host "FAILED: $example with exit code $exitCode" -ForegroundColor Red
        }
        else {
            $passedExamples += $example
        }

        Write-Host "Finished: $example (Duration: $($duration.ToString('hh\:mm\:ss')))"
        Write-Host "########################################"
    }
    else {
        Write-Host "SKIPPED (not found): $example" -ForegroundColor Yellow
    }
}

# =============================================================================
# Summary
# =============================================================================

Write-Host ""
Write-Host "========================================"
Write-Host "EXECUTION SUMMARY"
Write-Host "========================================"
Write-Host "Category: $Category"
Write-Host "Total examples run: $($selectedExamples.Count)"
Write-Host "Passed: $($passedExamples.Count)" -ForegroundColor Green
Write-Host "Failed: $($failedExamples.Count)" -ForegroundColor $(if ($failedExamples.Count -gt 0) { "Red" } else { "Green" })

if ($failedExamples.Count -gt 0) {
    Write-Host ""
    Write-Host "FAILED EXAMPLES:" -ForegroundColor Red
    foreach ($failed in $failedExamples) {
        Write-Host "  - $failed" -ForegroundColor Red
    }
    if ($logFile) {
        "" | Out-File -FilePath $logFile -Append -Encoding UTF8
        "========================================" | Out-File -FilePath $logFile -Append -Encoding UTF8
        "FAILED EXAMPLES:" | Out-File -FilePath $logFile -Append -Encoding UTF8
        foreach ($failed in $failedExamples) {
            "  - $failed" | Out-File -FilePath $logFile -Append -Encoding UTF8
        }
    }
    exit 1
}
else {
    Write-Host ""
    Write-Host "All examples passed successfully!" -ForegroundColor Green
}
