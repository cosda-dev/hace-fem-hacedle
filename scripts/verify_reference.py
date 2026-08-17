#!/usr/bin/env python3
"""Quick verify reference tensors"""

import numpy as np
from pathlib import Path

ref_dir = Path("golden/qwen2505b/block0")

for bin_file in ref_dir.glob("*.bin"):
    data = np.fromfile(bin_file, dtype=np.float32)
    print(f"{bin_file.name}: {data.shape}, mean={data.mean():.6}, std={data.std():.6}")