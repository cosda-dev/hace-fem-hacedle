#!/usr/bin/env pwsh
# GGUF Parity Test Runner
# Usage: .\run_parity_tests.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== A3-10 GGUF Parity Test Suite ===" -ForegroundColor Green
Write-Host ""

# Check for model
$modelPath = "D:\host\llama-models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
if (-not (Test-Path $modelPath)) {
    Write-Host "ERROR: Model not found at $modelPath" -ForegroundColor Red
    Write-Host "Available models:" -ForegroundColor Yellow
    Get-ChildItem -Path "D:\host\llama-models" -Filter "*.gguf" | Select-Object Name
    exit 1
}

Write-Host "Found model: $modelPath" -ForegroundColor Green
Write-Host ""

# Check for Rust toolchain
Write-Host "Step 1: Checking Rust toolchain..." -ForegroundColor Cyan
cargo --version | Out-String
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: cargo not found" -ForegroundColor Red
    exit 1
}

# Generate reference if needed
$refDir = "reference_output"
if (-not (Test-Path $refDir)) {
    Write-Host "Step 2: Generating reference data..." -ForegroundColor Cyan
    python scripts/generate_reference.py $modelPath $refDir
}

Write-Host ""
Write-Host "Step 3: Running parity tests in sequence..." -ForegroundColor Cyan
Write-Host ""

# Run tests SEQUENTIALLY - MUST pass in order
$tests = @(
    @{name="q4k_parity"; desc="Q4_K Dequant Parity"},
    @{name="rmsnorm_parity"; desc="RMSNorm Parity"},
    @{name="rope_parity"; desc="RoPE Parity"},
    @{name="block0_parity"; desc="Block0 Forward Parity"},
    @{name="kv_parity"; desc="KV Cache Parity"},
    @{name="logits_parity"; desc="Logits Parity"}
)

$failed = @()
foreach ($test in $tests) {
    Write-Host "Running $($test.desc)..." -ForegroundColor Yellow
    
    # Run test - it should handle model not found gracefully
    cargo test --test $($test.name) --features std -- --nocapture --test-threads=1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "PASS: $($test.name)" -ForegroundColor Green
    } else {
        Write-Host "FAIL: $($test.name)" -ForegroundColor Red
        $failed += $test.name
        
        # CRD Gate: STOP on Q4K or Block0 failure
        if ($test.name -in @("q4k_parity", "block0_parity")) {
            Write-Host ""
            Write-Host "CRD GATE FAILURE - STOPPING TEST SUITE" -ForegroundColor Red
            Write-Host "Cannot proceed without Q4K/Block0 parity!" -ForegroundColor Red
            exit 1
        }
    }
    Write-Host ""
}

Write-Host "=== Test Summary ===" -ForegroundColor Green
if ($failed.Count -eq 0) {
    Write-Host "All tests PASSED!" -ForegroundColor Green
    Write-Host "Ready for A3-10 End-to-End Parity Test" -ForegroundColor Green
} else {
    Write-Host "Failed tests: $($failed -join ', ')" -ForegroundColor Red
    exit 1
}