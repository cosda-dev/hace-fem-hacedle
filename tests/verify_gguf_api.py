from gguf import GGUFReader

reader = GGUFReader(r"D:\host\llama-models\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf")

print("=== GGUF Metadata ===")
print()

# List available attributes
print("Available keys:")
for k in sorted([k for k in dir(reader) if not k.startswith('_')])[:20]:
    print(f"  {k}")
print()

# Try to read metadata
if hasattr(reader, 'fields'):
    for key, val in reader.fields.items():
        print(f"{key} = {val}")