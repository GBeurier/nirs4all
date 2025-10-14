param(
    [int]$Index = 0,    # 1-based index to run a single example
    [int]$Start = 0     # 1-based start index to run all examples from
)

$examples = @(
    "Q1_regression.py",
    "Q1_classif.py",
    "Q2_multimodel.py",
    "Q3_finetune.py",
    "Q4_multidatasets.py",
    "Q5_predict.py",
    "Q6_multisource.py",
    "Q7_discretization.py",
    "Q8_shap.py",
    "Q9_acp_spread.py",
    "Q10_resampler.py"
)

# ## PARALLEL
# foreach ($example in $examples) {
#     if (Test-Path "$example") {
#         Start-Process -FilePath "python" -ArgumentList "$example" -NoNewWindow
#         Write-Host "Launch: $example"
#         Write-Host "########################################"
#         Write-Host "Finished running: $example"
#         Write-Host "########################################"
#     }
# }

# Determine which examples to run based on parameters
if ($Index -gt 0 -and $Start -gt 0) {
    Write-Host "Error: Specify only -Index OR -Start, not both." -ForegroundColor Red
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

# SEQUENTIAL
foreach ($example in $selectedExamples) {
    if (Test-Path "$example") {
        Start-Process -FilePath "python" -ArgumentList "$example" -NoNewWindow -Wait
        Write-Host "Launch: $example"
        Write-Host "########################################"
        Write-Host "Finished running: $example"
        Write-Host "########################################"
    }
}
