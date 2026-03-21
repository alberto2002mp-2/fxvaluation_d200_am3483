$ErrorActionPreference = "Stop"

$venvDir = "venv"

function Get-PythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @("py", "-3")
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @("python")
    }
    throw "Python 3 was not found in PATH. Install Python 3.10+ and rerun this script."
}

$pythonCmd = Get-PythonCommand
$pythonArgs = if ($pythonCmd.Length -gt 1) { $pythonCmd[1..($pythonCmd.Length - 1)] } else { @() }
$pythonVersion = & $pythonCmd[0] @pythonArgs --version 2>&1
$pythonExe = Join-Path $venvDir "Scripts\\python.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "FX Valuation Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Python interpreter: $pythonVersion" -ForegroundColor Green

if (Test-Path $venvDir) {
    Write-Host "Removing existing $venvDir environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venvDir
}

Write-Host "Creating virtual environment in $venvDir..." -ForegroundColor Yellow
& $pythonCmd[0] @pythonArgs -m venv $venvDir

Write-Host "Upgrading packaging tools..." -ForegroundColor Yellow
& $pythonExe -m pip install --upgrade pip setuptools wheel

Write-Host "Installing pinned repository requirements..." -ForegroundColor Yellow
& $pythonExe -m pip install -r requirements.txt

Write-Host "" 
Write-Host "Environment ready." -ForegroundColor Green
Write-Host "Activate with:" -ForegroundColor Yellow
Write-Host "  .\\venv\\Scripts\\Activate.ps1" -ForegroundColor White
