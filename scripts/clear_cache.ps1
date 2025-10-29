# Clear Python cache directories
# This script removes all __pycache__ directories recursively
# Note: __pycache__ directories contain .pyc bytecode files

Write-Host "Clearing Python cache directories..."

# Remove all __pycache__ directories
Get-ChildItem -Recurse -Directory -Filter __pycache__ | ForEach-Object {
    Write-Host "Removing directory: $($_.FullName)"
    Remove-Item $_.FullName -Recurse -Force
}

Write-Host "Cache clearing completed."