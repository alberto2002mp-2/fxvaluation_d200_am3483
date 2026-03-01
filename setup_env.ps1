# Data Science Environment Setup Script
# This script sets up a virtual environment and installs required dependencies

# Enable error handling
$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Data Science Project Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "Checking if Python is installed..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python from https://www.python.org/" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Check if venv already exists
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists. Deleting it..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv"
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv
Write-Host "Virtual environment created" -ForegroundColor Green
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
Write-Host "Virtual environment activated" -ForegroundColor Green
Write-Host ""

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel
Write-Host "pip upgraded" -ForegroundColor Green
Write-Host ""

# Install requirements
Write-Host "Installing requirements from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt
Write-Host "All requirements installed" -ForegroundColor Green
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Your virtual environment is now active!" -ForegroundColor Green
Write-Host ""
Write-Host "Project structure:" -ForegroundColor Yellow
Write-Host "  data/         - Store your datasets here" -ForegroundColor White
Write-Host "  notebooks/    - Jupyter notebooks for exploration and analysis" -ForegroundColor White
Write-Host "  src/          - Python modules and utilities" -ForegroundColor White
Write-Host ""
Write-Host "To deactivate the environment in the future, run:" -ForegroundColor Yellow
Write-Host "  deactivate" -ForegroundColor White
Write-Host ""
