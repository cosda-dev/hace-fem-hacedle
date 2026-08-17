# GGUF Metadata Verification Script
$modelPath = "D:\host\llama-models\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"
$ggufDump = "C:\Users\84334\AppData\Roaming\Python\Python314\Scripts\gguf-dump.exe"

if (Test-Path $ggufDump) {
    & $ggufDump $modelPath --metadata 2>&1 | Select-String -Pattern "general.architecture|qwen2.block_count|qwen2.attention.head_count|qwen2.embedding_length|qwen2.context_length|qwen2.attention.head_count_kv"
} else {
    Write-Host "gguf-dump.exe not found"
}