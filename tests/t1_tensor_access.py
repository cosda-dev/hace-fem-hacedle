import numpy as np
from gguf import GGUFReader

model_path = r"D:\host\llama-models\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"

reader = GGUFReader(model_path)

print("=== GGUF Reader Verification ===")

# Debug: list available attributes
print("Available attributes:")
for attr in dir(reader):
    if not attr.startswith('_'):
        print(f"  {attr}")

print()

# List first few tensors
print("First 5 tensors:")
for i, tensor in enumerate(reader.tensors[:5]):
    name = tensor.name
    shape = tensor.shape
    dtype = tensor.dtype
    print(f"  {name}: shape={shape}, dtype={dtype}")