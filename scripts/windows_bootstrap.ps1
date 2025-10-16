# OTREP-X PRIME Windows Bootstrap Script
# Version: 1.0.0
# Requires: PowerShell 5.1+

Write-Host "OTREP-X PRIME System Initialization" -ForegroundColor Cyan

# Validate execution policy
if ((Get-ExecutionPolicy) -gt 'RemoteSigned') {
    Write-Warning "Setting execution policy to RemoteSigned"
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
}

# Ensure Chocolatey package manager
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Chocolatey package manager..." -ForegroundColor Yellow
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
}

# Install base dependencies
$packages = @(
    'git',
    'python',
    'docker-desktop',
    'vcredist-all'
)

foreach ($package in $packages) {
    choco install $package -y --no-progress
}

# Create project directory structure
$directories = @(
    'config',
    'data/input',
    'data/output',
    'logs',
    'src/modules',
    'tests'
)

foreach ($dir in $directories) {
    New-Item -Path $dir -ItemType Directory -Force | Out-Null
}

Write-Host "Bootstrap complete. System ready for deployment." -ForegroundColor Green
