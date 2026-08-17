@echo off
"C:\Users\84334\AppData\Roaming\Python\Python314\Scripts\gguf-dump.exe" "D:\host\llama-models\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf" --metadata > "%~dp0gguf_metadata.txt" 2>&1
type "%~dp0gguf_metadata.txt" | findstr /C:"general.architecture" /C:"qwen2.block_count" /C:"qwen2.attention.head_count" /C:"qwen2.embedding_length"