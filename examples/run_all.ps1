$examples = @("Q1.py", "Q1_classif.py", "Q1_finetune.py", "Q2.py", "Q3.py", "Q4.py", "Q5.py", "Q7.py")
foreach ($example in $examples) {
    if (Test-Path "examples/$example") {
        Start-Process -FilePath "python" -ArgumentList "examples/$example" -NoNewWindow
        Write-Host "Lancé: $example"
    }
}