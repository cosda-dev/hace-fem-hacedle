#!/usr/bin/env pwsh
# Golden Bundle Generator for A3-10
# Creates standardized test data for regression testing

param(
    [Parameter(Mandatory=$false)]
    [string]$Model = "qwen25-0.5b",
    
    [Parameter(Mandatory=$false)]
    [string]$Prompt = "Hello"
)

$goldenDir = "golden/$Model"
$manifestFile = "$goldenDir/manifest.json"

Write-Host "=== Creating Golden Bundle: $Model ===" -ForegroundColor Green

# Create directory structure
New-Item -ItemType Directory -Path "$goldenDir/block0" -Force | Out-Null
New-Item -ItemType Directory -Path "$goldenDir/block1" -Force | Out-Null

# Create manifest
@{
    model = $Model
    version = "1.0"
    prompt = $Prompt
    created = Get-Date -Format "yyyy-MM-dd"
    files = @{
        model_spec = "model_spec.sio"
        block0_q = "block0/q.bin"
        block0_k = "block0/k.bin"
        block0_v = "block0/v.bin"
        block0_attn = "block0/attn.bin"
        block0_ffn = "block0/ffn.bin"
        block0_out = "block0/out.bin"
        logits = "logits.bin"
    }
} | ConvertTo-Json | Set-Content $manifestFile

Write-Host "Created: $goldenDir" -ForegroundColor Green
Write-Host "Run llama.cpp dump to populate tensor files" -ForegroundColor Yellow