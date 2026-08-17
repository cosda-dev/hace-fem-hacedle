import torch
import numpy as np

def rmsnorm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x) + eps)
    return x / rms * weight

x = torch.randn(256, dtype=torch.float32)
weight = torch.randn(256, dtype=torch.float32)
out = rmsnorm(x, weight)
out.numpy().tofile("rmsnorm_golden.bin")
print("Generated rmsnorm_golden.bin")

