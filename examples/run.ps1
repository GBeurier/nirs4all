param(
    [Alias('i')]
    [int]$Index = 0,       # 1-based index to run a single example

    [Alias('b')]
    [int]$Begin = 0,       # 1-based start index to run all examples from

    [Alias('e')]
    [int]$End = 0,       # 1-based end index to run all examples to

    [Alias('n')]
    [string]$Name = "",    # Name of a single example to run (e.g., "Q1_classif.py")

    [Alias('l')]
    [switch]$Log,          # Enable logging to log.txt

    [Alias('p')]
    [switch]$Plot,          # Pass boolean option to all examples (enables plots)

    [Alias('s')]
    [switch]$Show          # Pass boolean option to all examples (enables plots)
)

$examples = @(
    "Q1_classif.py",
    "Q1_regression.py",
    "Q2_groupsplit.py",
    "Q2_multimodel.py",
    "Q2B_force_group.py",
    "Q3_finetune.py",
    "Q4_multidatasets.py",
    "Q5_predict.py",
    "Q5_predict_NN.py",
    "Q6_multisource.py",
    "Q7_discretization.py",
    "Q8_shap.py",
    "Q9_acp_spread.py",
    "Q10_resampler.py",
    "Q11_flexible_inputs.py",
    "Q12_sample_augmentation.py",
    "Q13_nm_headers.py",
    "Q14_workspace.py",
    "Q15_jax_models.py",
    "Q16_pytorch_models.py",
    "Q17_nicon_comparison.py",
    "Q18_stacking.py",
    "Q19_pls_methods.py",
    "Q21_feature_selection.py",
    "Q22_concat_transform.py",
    "Q23_generator_syntax.py",
    "Q23b_generator.py",
    "Q24_generator_advanced.py",
    "Q25_complex_pipeline_pls.py",
    "Q26_nested_or_preprocessing.py",
    "Q27_transfer_analysis.py",
    "Q28_sample_filtering.py",
    "Q29_signal_conversion.py",
    "Q30_branching.py",
    "Q31_outlier_branching.py",
    "Q32_export_bundle.py",
    "Q33_retrain_transfer.py",
    "Q34_aggregation.py",
    "Q35_metadata_branching.py",
    "Q36_repetition_transform.py",
    "Q_meta_stacking.py",
    "Q_complex_all_keywords.py",
    "Q_feature_augmentation_modes.py",
    "Q_merge_branches.py",
    "Q_merge_sources.py",
    "baseline_sota.py",
    "X0_pipeline_sample.py",
    "X1_metadata.py",
    "X2_sample_augmentation.py",
    "X3_hiba_full.py",
    "X4_features.py"
)

# Setup logging if requested
$logFile = $null
if ($Log) {
    $logFile = Join-Path (Get-Location) "log.txt"
    Write-Host "Logging enabled: $logFile" -ForegroundColor Green
    # Initialize log file with header
    "=================================================" | Out-File -FilePath $logFile -Encoding UTF8
    "Log started at: $(Get-Date)" | Out-File -FilePath $logFile -Append -Encoding UTF8
    "=================================================" | Out-File -FilePath $logFile -Append -Encoding UTF8
    "" | Out-File -FilePath $logFile -Append -Encoding UTF8
}

# Determine which examples to run based on parameters
$paramCount = 0
if ($Index -gt 0) { $paramCount++ }
if ($Begin -gt 0) { $paramCount++ }
if ($End -gt 0) { $paramCount++ }
if ($Name -ne "") { $paramCount++ }

if ($paramCount -gt 1) {
    Write-Host "Error: Specify only one of -Index, -Begin, or -Name." -ForegroundColor Red
    exit 1
}

$selectedExamples = $examples
if ($Index -gt 0) {
    if ($Index -lt 1 -or $Index -gt $examples.Count) {
        Write-Host ("Error: Index {0} is out of range. Valid range is 1..{1}." -f $Index, $examples.Count) -ForegroundColor Red
        exit 1
    }
    $selectedExamples = @($examples[$Index - 1])
    Write-Host ("Running single example #{0}: {1}" -f $Index, $selectedExamples[0])
}
elseif ($Begin -gt 0) {
    if ($Begin -lt 1 -or $Begin -gt $examples.Count) {
        Write-Host ("Error: Start index {0} is out of range. Valid range is 1..{1}." -f $Begin, $examples.Count) -ForegroundColor Red
        exit 1
    }
    $startIndex = $Begin - 1
    $selectedExamples = $examples[$startIndex..($examples.Count - 1)]
    Write-Host ("Running all examples starting from #{0} (count: {1})" -f $Begin, $selectedExamples.Count)
}
elseif ($Name -ne "") {
    # Find the example by name (case-insensitive)
    $matchedExample = $examples | Where-Object { $_ -like $Name }
    if (-not $matchedExample) {
        Write-Host "Error: Example '$Name' not found in the list." -ForegroundColor Red
        Write-Host "Available examples:" -ForegroundColor Yellow
        $examples | ForEach-Object { Write-Host "  $_" }
        exit 1
    }
    $selectedExamples = @($matchedExample)
    Write-Host "Running example by name: $matchedExample"
}

# Disable emojis only when logging to avoid encoding issues with file output
if ($Log) {
    $env:DISABLE_EMOJI = "1"
    Write-Host "Emojis disabled for logging" -ForegroundColor Yellow
}
else {
    Remove-Item Env:\DISABLE_EMOJI -ErrorAction SilentlyContinue
}

# Track failed examples
$failedExamples = @()

# SEQUENTIAL
foreach ($example in $selectedExamples) {
    if (Test-Path "$example") {
        $startTime = Get-Date
        Write-Host "########################################"
        Write-Host "Launch: $example"
        Write-Host "########################################"

        $args = @()
        if ($Plot) { $args += "--plots" }
        if ($Show) { $args += "--show" }

        $exitCode = 0
        try {
            if ($logFile) {
                # Log the header
                "===============================================" | Out-File -FilePath $logFile -Append -Encoding UTF8
                "Starting: $example at $startTime" | Out-File -FilePath $logFile -Append -Encoding UTF8
                "===============================================" | Out-File -FilePath $logFile -Append -Encoding UTF8

                & python $example @args 2>&1 | Tee-Object -FilePath $logFile -Append
                $exitCode = $LASTEXITCODE

                # Log the footer
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

        # Track failed examples
        if ($exitCode -ne 0) {
            $failedExamples += "$example (exit code: $exitCode)"
            Write-Host "FAILED: $example with exit code $exitCode" -ForegroundColor Red
        }

        $endTime = Get-Date
        Write-Host "Finished running: $example (Duration: $(($endTime - $startTime).ToString('hh\:mm\:ss')))"
        Write-Host "########################################"
    }
}

# Summary of failed examples
Write-Host ""
Write-Host "========================================"
Write-Host "EXECUTION SUMMARY"
Write-Host "========================================"
Write-Host "Total examples run: $($selectedExamples.Count)"
Write-Host "Failed: $($failedExamples.Count)" -ForegroundColor $(if ($failedExamples.Count -gt 0) { "Red" } else { "Green" })
Write-Host "Passed: $($selectedExamples.Count - $failedExamples.Count)" -ForegroundColor Green

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