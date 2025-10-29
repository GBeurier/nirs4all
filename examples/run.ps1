param(
    [Alias('i')]
    [int]$Index = 0,       # 1-based index to run a single example

    [Alias('s')]
    [int]$Start = 0,       # 1-based start index to run all examples from

    [Alias('n')]
    [string]$Name = "",    # Name of a single example to run (e.g., "Q1_classif.py")

    [Alias('l')]
    [switch]$Log,          # Enable logging to log.txt

    [Alias('p')]
    [switch]$Plot          # Pass boolean option to all examples (enables plots)
)

$examples = @(
    "Q1_classif.py",
    "Q1_classif_tf.py",
    "Q1_groupsplit.py",
    "Q1_regression.py",
    "Q2_multimodel.py",
    "Q3_finetune.py",
    "Q4_multidatasets.py",
    "Q5_predict_NN.py",
    "Q5_predict.py",
    "Q6_multisource.py",
    "Q7_discretization.py",
    "Q8_shap.py",
    "Q9_acp_spread.py",
    "Q10_resampler.py",
    "Q11_flexible_inputs.py",
    "Q12_sample_augmentation.py",
    "Q13_nm_headers.py",
    "Q14_workspace.py",
    "X1_metadata.py",
    "X2_sample_augmentation.py",
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

# ## PARALLEL
# foreach ($example in $examples) {
#     if (Test-Path "$example") {
#         Start-Process -FilePath "python" -ArgumentList "$example" -NoNewWindow
#         Write-Output-Dual "Launch: $example"
#         Write-Output-Dual "########################################"
#         Write-Output-Dual "Finished running: $example"
#         Write-Output-Dual "########################################"
#     }
# }

# Determine which examples to run based on parameters
$paramCount = 0
if ($Index -gt 0) { $paramCount++ }
if ($Start -gt 0) { $paramCount++ }
if ($Name -ne "") { $paramCount++ }

if ($paramCount -gt 1) {
    Write-Host "Error: Specify only one of -Index, -Start, or -Name." -ForegroundColor Red
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
elseif ($Start -gt 0) {
    if ($Start -lt 1 -or $Start -gt $examples.Count) {
        Write-Host ("Error: Start index {0} is out of range. Valid range is 1..{1}." -f $Start, $examples.Count) -ForegroundColor Red
        exit 1
    }
    $startIndex = $Start - 1
    $selectedExamples = $examples[$startIndex..($examples.Count - 1)]
    Write-Host ("Running all examples starting from #{0} (count: {1})" -f $Start, $selectedExamples.Count)
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

# SEQUENTIAL
foreach ($example in $selectedExamples) {
    if (Test-Path "$example") {
        $startTime = Get-Date
        Write-Host "Launch: $example"
        Write-Host "########################################"

        if ($logFile) {
            # Log the header
            "===============================================" | Out-File -FilePath $logFile -Append -Encoding UTF8
            "Starting: $example at $startTime" | Out-File -FilePath $logFile -Append -Encoding UTF8
            "===============================================" | Out-File -FilePath $logFile -Append -Encoding UTF8

            # Run and capture output using Tee-Object to show AND log
            if ($Plot) {
                & python $example --show-plots 2>&1 | Tee-Object -FilePath $logFile -Append
            }
            else {
                & python $example 2>&1 | Tee-Object -FilePath $logFile -Append
            }

            # Log the footer
            "Finished: $example at $(Get-Date)" | Out-File -FilePath $logFile -Append -Encoding UTF8
            "" | Out-File -FilePath $logFile -Append -Encoding UTF8
        }
        else {
            if ($Plot) {
                & python $example --show-plots
            }
            else {
                & python $example
            }
        }

        $endTime = Get-Date
        Write-Host "Finished running: $example (Duration: $(($endTime - $startTime).ToString('hh\:mm\:ss')))"
        Write-Host "########################################"
    }
}