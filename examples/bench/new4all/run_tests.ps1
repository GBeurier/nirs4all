# This array lives in the script scope and will collect all failures.
$script:failedScripts = @()

function Run-PythonScripts {
    param([string]$pattern)

    # For each file matching the pattern:
    Get-ChildItem -Filter $pattern | ForEach-Object {
        $fileName = $_.Name
        Write-Host "`nRunning $fileName" -ForegroundColor Cyan

        try {
            # Launch python and check its exit code
            python $_.FullName
            if ($LASTEXITCODE -ne 0) {
                throw "Python exited with code $LASTEXITCODE"
            }
        }
        catch {
            # Record the failure into the script-scoped array
            Write-Host "❌ Failed: $fileName — $($_.Exception.Message)" -ForegroundColor Red
            $script:failedScripts += $fileName
        }
    }
}

# Run all debug*.py and test*.py
Run-PythonScripts -pattern "debug*.py"
Run-PythonScripts -pattern "test*.py"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
Write-Host "`n=== Summary ===" -ForegroundColor Yellow

if ($script:failedScripts.Count -eq 0) {
    Write-Host "✅ All scripts ran successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ The following scripts failed:" -ForegroundColor Red
    foreach ($f in $script:failedScripts) {
        Write-Host " - $f" -ForegroundColor Red
    }
}
