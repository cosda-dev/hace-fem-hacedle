$modelPath = "D:\host\llama-models\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"

if (-not (Test-Path $modelPath)) {
    Write-Host "ERROR: Model file not found"
    exit 1
}

$bytes = [System.IO.File]::ReadAllBytes($modelPath)

# Helper: read little-endian u64 from byte array at position
function Get-UInt64At {
    param([byte[]]$b, [int]$pos)
    return [System.BitConverter]::ToUInt64($b, $pos)
}

function Get-UInt32At {
    param([byte[]]$b, [int]$pos)
    return [System.BitConverter]::ToUInt32($b, $pos)
}

# Header
$magic = [System.Text.Encoding]::ASCII.GetString($bytes, 0, 4)
if ($magic -ne "GGUF") {
    Write-Host "ERROR: Invalid magic: $magic"
    exit 1
}

$version = Get-UInt32At $bytes 4
Write-Host "GGUF Version: $version"

$tensorCount = Get-UInt64At $bytes 12
Write-Host "Tensor count: $tensorCount"

$kvCount = Get-UInt64At $bytes 20
Write-Host "Metadata entries: $kvCount"

# Parse metadata
$pos = 28
$metadata = @{}

for ($i = 0; $i -lt $kvCount; $i++) {
    $keyLen = Get-UInt64At $bytes $pos; $pos += 8
    $key = [System.Text.Encoding]::UTF8.GetString($bytes, $pos, [int]$keyLen); $pos += [int]$keyLen
    
    $typeByte = $bytes[$pos]; $pos += 1
    
    if ($typeByte -eq 2) {
        $strLen = Get-UInt64At $bytes $pos; $pos += 8
        $value = [System.Text.Encoding]::UTF8.GetString($bytes, $pos, [int]$strLen); $pos += [int]$strLen
        $metadata[$key] = $value
    } else {
        if ($typeByte -eq 1) { $pos += 8 }
        elseif ($typeByte -eq 5) { $pos += 4 }
    }
}

Write-Host ""
Write-Host "=== GGUF Metadata Golden Test ==="
Write-Host ""

$expected = @{
    "general.architecture" = "qwen2"
    "qwen2.block_count" = "28"
    "qwen2.attention.head_count" = "12"
}

$passed = 0
$failures = @()

foreach ($key in $expected.Keys) {
    if ($metadata.ContainsKey($key)) {
        $val = $metadata[$key]
        $exp = $expected[$key]
        if ($val -eq $exp) {
            Write-Host "✅ $key = $val"
            $passed++
        } else {
            Write-Host "❌ $key = $val (expected: $exp)"
            $failures += $key
        }
    } else {
        Write-Host "❌ $key NOT FOUND"
        $failures += $key
    }
}

Write-Host ""
Write-Host "=== Summary: $passed/$($expected.Count) passed ==="

if ($failures.Count -eq 0) {
    Write-Host "✅ TEST PASSED"
    exit 0
} else {
    Write-Host "Failures: $($failures -join ', ')"
    exit 1
}