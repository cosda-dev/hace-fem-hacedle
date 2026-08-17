# Simple GGUF Verification - Direct byte access
$modelPath = "D:\host\llama-models\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"

if (-not (Test-Path $modelPath)) {
    Write-Host "ERROR: File not found"
    exit 1
}

$bytes = [System.IO.File]::ReadAllBytes($modelPath)

# Helper
function ReadU32([int]$pos) {
    return [System.BitConverter]::ToInt32($bytes, $pos)
}

function ReadU64([int]$pos) {
    # Manual little-endian read
    return ($bytes[$pos]) -bor ($bytes[$pos+1] -shl 8) -bor ($bytes[$pos+2] -shl 16) -bor ($bytes[$pos+3] -shl 24)
}

function ReadStringAt([int]$pos, [int]$len) {
    return [System.Text.Encoding]::ASCII.GetString($bytes, $pos, $len)
}

# Header
$magic = ReadStringAt 0 4
Write-Host "Magic: $magic"

$version = ReadU32 4
Write-Host "Version: $version"

$tensorCount = ReadU64 12
Write-Host "Tensor count: $tensorCount"

$kvCount = ReadU64 20
Write-Host "Metadata entries: $kvCount"

Write-Host ""
Write-Host "=== Searching for Qwen2 metadata ==="

# Search for specific keys in raw bytes
$needle1 = [System.Text.Encoding]::ASCII.GetBytes("general.architecture")
$needle2 = [System.Text.Encoding]::ASCII.GetBytes("qwen2.block_count")
$needle3 = [System.Text.Encoding]::ASCII.GetBytes("qwen2.attention.head_count")

function FindKey($keyBytes) {
    $idx = [System.Array]::IndexOf($bytes, $keyBytes[0])
    for ($i = 0; $i -lt $bytes.Length - $keyBytes.Length; $i++) {
        $match = $true
        for ($j = 0; $j -lt $keyBytes.Length; $j++) {
            if ($bytes[$i + $j] -ne $keyBytes[$j]) { $match = $false; break }
        }
        if ($match) { return $i }
    }
    return -1
}

$pos1 = FindKey $needle1
if ($pos1 -ge 0) {
    $valPos = $pos1 + $needle1.Length + 1
    $strLen = ReadU64($valPos + 8)
    $val = ReadStringAt ($valPos + 16) $strLen
    Write-Host "general.architecture = $val"
}

$pos2 = FindKey $needle2
if ($pos2 -ge 0) {
    $valPos = $pos2 + $needle2.Length + 1
    $val = ReadU64($valPos + 8)
    Write-Host "qwen2.block_count = $val"
}

$pos3 = FindKey $needle3
if ($pos3 -ge 0) {
    $valPos = $pos3 + $needle3.Length + 1
    $val = ReadU64($valPos + 8)
    Write-Host "qwen2.attention.head_count = $val"
}

Write-Host ""
Write-Host "✅ Verification complete - check values above"