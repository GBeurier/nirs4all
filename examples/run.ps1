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
    Category: user, developer, reference, legacy, all (default: all)

.PARAMETER Log
    Enable logging to log.txt

.PARAMETER Plot
    Generate plots

.PARAMETER Show
    Show plots interactively

.PARAMETER Quick
    Quick mode: skip deep learning examples

.EXAMPLE
    ./run.ps1                     # Run all examples
    ./run.ps1 -c user             # Run only User path examples
    ./run.ps1 -c legacy           # Run only legacy Q*/X* examples
    ./run.ps1 -i 1                # Run first example
    ./run.ps1 -n "U01*.py"        # Run by name pattern
    ./run.ps1 -q                  # Skip deep learning examples
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
    [switch]$Show,

    [Alias('q')]
    [switch]$Quick
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
    "user/02_data_handling/U05_flexible_inputs.py"
    "user/02_data_handling/U06_multi_datasets.py"
    "user/02_data_handling/U07_multi_source.py"
    "user/02_data_handling/U08_wavelength_handling.py"
    # 03_preprocessing
    "user/03_preprocessing/U09_preprocessing_basics.py"
    "user/03_preprocessing/U10_feature_augmentation.py"
    "user/03_preprocessing/U11_sample_augmentation.py"
    "user/03_preprocessing/U12_signal_conversion.py"
    # 04_models
    "user/04_models/U13_multi_model.py"
    "user/04_models/U14_hyperparameter_tuning.py"
    "user/04_models/U15_stacking_ensembles.py"
    "user/04_models/U16_pls_variants.py"
    # 05_cross_validation
    "user/05_cross_validation/U17_cv_strategies.py"
    "user/05_cross_validation/U18_group_splitting.py"
    "user/05_cross_validation/U19_sample_filtering.py"
    "user/05_cross_validation/U20_aggregation.py"
    # 06_deployment
    "user/06_deployment/U21_save_load_predict.py"
    "user/06_deployment/U22_export_bundle.py"
    "user/06_deployment/U23_workspace_management.py"
    "user/06_deployment/U24_sklearn_integration.py"
    # 07_explainability
    "user/07_explainability/U25_shap_basics.py"
    "user/07_explainability/U26_shap_sklearn.py"
    "user/07_explainability/U27_feature_selection.py"
)

# Developer path examples (new structure)
$developerExamples = @(
    # 01_advanced_pipelines
    "developer/01_advanced_pipelines/D01_branching_basics.py"
    "developer/01_advanced_pipelines/D02_branching_advanced.py"
    "developer/01_advanced_pipelines/D03_merge_strategies.py"
    "developer/01_advanced_pipelines/D04_source_branching.py"
    "developer/01_advanced_pipelines/D05_meta_stacking.py"
    # 02_generators
    "developer/02_generators/D06_generator_basics.py"
    "developer/02_generators/D07_generator_advanced.py"
    "developer/02_generators/D08_generator_nested.py"
    "developer/02_generators/D09_constraints_presets.py"
    # 03_deep_learning
    "developer/03_deep_learning/D10_nicon_tensorflow.py"
    "developer/03_deep_learning/D11_pytorch_models.py"
    "developer/03_deep_learning/D12_jax_models.py"
    "developer/03_deep_learning/D13_framework_comparison.py"
    # 04_transfer_learning
    "developer/04_transfer_learning/D14_retrain_modes.py"
    "developer/04_transfer_learning/D15_transfer_analysis.py"
    "developer/04_transfer_learning/D16_domain_adaptation.py"
    # 05_advanced_features
    "developer/05_advanced_features/D17_outlier_partitioning.py"
    "developer/05_advanced_features/D18_metadata_branching.py"
    "developer/05_advanced_features/D19_repetition_transform.py"
    "developer/05_advanced_features/D20_concat_transform.py"
    # 06_internals
    "developer/06_internals/D21_session_workflow.py"
    "developer/06_internals/D22_custom_controllers.py"
)

# Reference examples (new structure)
$referenceExamples = @(
    "reference/R01_pipeline_syntax.py"
    "reference/R02_generator_reference.py"
    "reference/R03_all_keywords.py"
    "reference/R04_legacy_api.py"
)

# Legacy examples (old Q*/X* structure - for transition period)
$legacyExamples = @(
    "Q1_classif.py"
    "Q1_regression.py"
    "Q2_groupsplit.py"
    "Q2_multimodel.py"
    "Q2B_force_group.py"
    "Q3_finetune.py"
    "Q4_multidatasets.py"
    "Q5_predict.py"
    "Q5_predict_NN.py"
    "Q6_multisource.py"
    "Q7_discretization.py"
    "Q8_shap.py"
    "Q9_acp_spread.py"
    "Q10_resampler.py"
    "Q11_flexible_inputs.py"
    "Q12_sample_augmentation.py"
    "Q13_nm_headers.py"
    "Q14_workspace.py"
    "Q15_jax_models.py"
    "Q16_pytorch_models.py"
    "Q17_nicon_comparison.py"
    "Q18_stacking.py"
    "Q19_pls_methods.py"
    "Q21_feature_selection.py"
    "Q22_concat_transform.py"
    "Q23_generator_syntax.py"
    "Q23b_generator.py"
    "Q24_generator_advanced.py"
    "Q25_complex_pipeline_pls.py"
    "Q26_nested_or_preprocessing.py"
    "Q27_transfer_analysis.py"
    "Q28_sample_filtering.py"
    "Q29_signal_conversion.py"
    "Q30_branching.py"
    "Q31_outlier_branching.py"
    "Q32_export_bundle.py"
    "Q33_retrain_transfer.py"
    "Q34_aggregation.py"
    "Q35_metadata_branching.py"
    "Q36_repetition_transform.py"
    "Q40_new_api.py"
    "Q41_sklearn_shap.py"
    "Q42_session_workflow.py"
    "Q_meta_stacking.py"
    "Q_complex_all_keywords.py"
    "Q_feature_augmentation_modes.py"
    "Q_merge_branches.py"
    "Q_merge_sources.py"
    "Q_sklearn_wrapper.py"
    "X0_pipeline_sample.py"
    "X1_metadata.py"
    "X2_sample_augmentation.py"
    "X4_features.py"
)

# Deep learning examples to skip in quick mode (basenames)
$dlExamplesPatterns = @(
    "D10_nicon_tensorflow.py"
    "D11_pytorch_models.py"
    "D12_jax_models.py"
    "D13_framework_comparison.py"
    "Q15_jax_models.py"
    "Q16_pytorch_models.py"
    "Q17_nicon_comparison.py"
    "Q5_predict_NN.py"
    "X3_hiba_full.py"
)

# =============================================================================
# Build Examples List
# =============================================================================

function Is-DLExample {
    param([string]$example)
    $basename = Split-Path -Leaf $example
    return $dlExamplesPatterns -contains $basename
}

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
    "legacy" {
        $examples = $legacyExamples
    }
    "all" {
        $examples = $userExamples + $developerExamples + $referenceExamples + $legacyExamples
    }
    default {
        Write-Host "Error: Unknown category '$Category'. Valid: user, developer, reference, legacy, all" -ForegroundColor Red
        exit 1
    }
}

# Filter to existing files only
$selectedExamples = $examples | Where-Object { Test-Path $_ }

# Apply quick mode filter
if ($Quick) {
    $selectedExamples = $selectedExamples | Where-Object { -not (Is-DLExample $_) }
    Write-Host "Quick mode: Skipping deep learning examples" -ForegroundColor Yellow
}

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
    "Quick mode: $Quick" | Out-File -FilePath $logFile -Append -Encoding UTF8
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
