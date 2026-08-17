#!/usr/bin/env pwsh
# Dump reference activations from llama.cpp
# Requires patched llama.cpp with --dump-activations support

param(
    [Parameter(Mandatory=$false)]
    [string]$ModelPath = "D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
    
    [Parameter(Mandatory=$false)]
    [string]$OutputDir = "reference_dump",
    
    [Parameter(Mandatory=$false)]
    [string]$Prompt = "Hello",
    
    [Parameter(Mandatory=$false)]
    [int]$Tokens = 1
)

Write-Host "=== llama.cpp Activation Dump ===" -ForegroundColor Cyan
Write-Host "Model: $ModelPath"
Write-Host "Output: $OutputDir"
Write-Host "Prompt: $Prompt"
Write-Host ""

# Check if model exists
if (-not (Test-Path $ModelPath)) {
    Write-Host "ERROR: Model not found. Available models:" -ForegroundColor Red
    Get-ChildItem -Path "D:/host/llama-models" -Filter "*.gguf" | Select-Object Name
    exit 1
}

# Create output directory
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# Check for llama-cli
$llamaCli = $null
$searchPaths = @(
    "D:/host/llama.cpp/build/bin/Release/llama-cli.exe",
    "D:/host/llama.cpp/build/bin/Debug/llama-cli.exe",
    "D:/host/llama.cpp/llama-cli.exe"
)

foreach ($path in $searchPaths) {
    if (Test-Path $path) {
        $llamaCli = $path
        break
    }
}

if ($null -eq $llamaCli) {
    Write-Host "WARNING: llama-cli.exe not found. Please build llama.cpp first." -ForegroundColor Yellow
    Write-Host "Or apply the patches in llama_cpp_dump_patches.md to add --dump-activations support"
    exit 1
}

Write-Host "Running llama.cpp with activation dump..." -ForegroundColor Yellow

# Run with dump
& $llamaCli @("--model", $ModelPath, "--dump-activations", $OutputDir, "-p", $Prompt, "-n", $Tokens.ToString())

Write-Host ""
Write-Host "=== Dump Complete ===" -ForegroundColor Green
Write-Host "Reference files in: $OutputDir" -ForegroundColor Green
Get-ChildItem -Path $OutputDir -ErrorAction SilentlyContinue | Select-Object Name