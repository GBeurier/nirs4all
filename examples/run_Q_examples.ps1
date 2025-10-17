param(
    [switch]$OutputFile,
    [string]$LogPath = "run_Q_output.log"
)

# Set environment variable to disable emoji
$env:DISABLE_EMOJI = "1"

$examples = @(
    "Q1_groupsplit.py",
    "Q1_regression.py",
    "Q1_classif.py",
    "Q2_multimodel.py",
    "Q3_finetune.py",
    "Q4_multidatasets.py",
    "Q5_predict.py",
    "Q5_predict_NN.py",
    "Q6_multisource.py",
    "Q7_discretization.py",
    "Q8_shap.py",
    "Q9_data_analysis.py",
    "Q10_resampler.py",
    "Q11_flexible_inputs.py",
    "Q12_sample_augmentation.py"
)

# Function to write output to console and optionally to file
function Write-Output-Dual {
    param([string]$Message)
    Write-Host $Message
    if ($OutputFile) {
        Add-Content -Path $LogPath -Value $Message
    }
}

# Initialize log file if output to file is enabled
if ($OutputFile) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "=== Q Examples Run Started: $timestamp ===" | Set-Content -Path $LogPath
    Write-Host "Output will be saved to: $LogPath"
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

# SEQUENTIAL
foreach ($example in $examples) {
    if (Test-Path "$example") {
        Write-Output-Dual "Launch: $example"
        Write-Output-Dual "########################################"

        if ($OutputFile) {
            # Capture full output including stdout and stderr to file
            & python "$example" 2>&1 | Tee-Object -FilePath $LogPath -Append | Write-Host
        } else {
            # Just run normally without capturing
            & python "$example"
        }

        Write-Output-Dual "########################################"
        Write-Output-Dual "Finished running: $example"
        Write-Output-Dual "########################################"
    }
}

# Print completion message
if ($OutputFile) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Output-Dual "=== Q Examples Run Completed: $timestamp ==="
}
