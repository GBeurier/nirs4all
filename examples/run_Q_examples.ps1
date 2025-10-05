## PARALLEL
$examples = @("Q1.py", "Q1_classif.py", "Q2.5_finetune.py", "Q2.py", "Q3.py", "Q4.py", "Q5.py", "Q7.py")
foreach ($example in $examples) {
    if (Test-Path "$example") {
        Start-Process -FilePath "python" -ArgumentList "examples/$example" -NoNewWindow
        Write-Host "Lancé: $example"
    }
}

## SEQUENTIAL
# $examples = @("Q1.py", "Q1_classif.py", "Q2.5_finetune.py", "Q2.py", "Q3.py", "Q4.py", "Q5.py", "Q7.py")
# foreach ($example in $examples) {
#     if (Test-Path "$example") {
#         Start-Process -FilePath "python" -ArgumentList "examples/$example" -NoNewWindow -Wait
#         Write-Host "Lancé: $example"
#     }
# }