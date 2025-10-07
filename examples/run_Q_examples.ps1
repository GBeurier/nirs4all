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
    "Q9_data_analysis.py",
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

# SEQUENTIAL
foreach ($example in $examples) {
    if (Test-Path "$example") {
        Start-Process -FilePath "python" -ArgumentList "$example" -NoNewWindow -Wait
        Write-Host "Launch: $example"
        Write-Host "########################################"
        Write-Host "Finished running: $example"
        Write-Host "########################################"
    }
}
