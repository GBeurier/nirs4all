$examples = @(
    "Q1_regression.py",
    "Q1_classif.py",
    "Q2_multimodel.py",
    "Q3_finetune.py",
    "Q4_multidatasets.py",
    "Q5_predict.py",
    "Q6_multisource.py",
    "Q7_discretization.py"
)

## PARALLEL
foreach ($example in $examples) {
    if (Test-Path "$example") {
        Start-Process -FilePath "python" -ArgumentList "$example" -NoNewWindow
        Write-Host "Lancé: $example"
    }
}

## SEQUENTIAL
# foreach ($example in $examples) {
#     if (Test-Path "$example") {
#         Start-Process -FilePath "python" -ArgumentList "examples/$example" -NoNewWindow -Wait
#         Write-Host "Lancé: $example"
#     }
# }