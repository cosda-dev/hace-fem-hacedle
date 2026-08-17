#!/usr/bin/env python3
"""P4: Mini Generation Parity - Greedy decode for Hello prompt"""

import numpy as np
from pathlib import Path
import json

def greedy_decode(logits: np.ndarray) -> int:
    """Greedy decoding: argmax"""
    return int(np.argmax(logits))

def main():
    # Simulate multi-token generation
    out = Path("golden/generation")
    out.mkdir(parents=True, exist_ok=True)
    
    # Load logits
    logits = np.frombuffer(
        Path("golden/logits/logits.bin").read_bytes(),
        dtype=np.float32
    )
    
    # Generation loop (greedy, 5 tokens)
    tokens = [9707]  # start with "hello"
    generated = []
    
    for step in range(5):
        # In real: run model forward, get logits
        next_token = greedy_decode(logits)
        generated.append(next_token)
        tokens.append(next_token)
        
        # Update logits for next step (simulated)
        logits = logits + np.random.randn(*logits.shape).astype(np.float32) * 0.001
    
    # Save
    result = {
        "prompt_tokens": tokens[:-5],
        "generated_tokens": generated,
        "generation_method": "greedy",
        "temperature": 0.0,
    }
    (out / "hello_generation.json").write_text(json.dumps(result, indent=2))
    
    print(f"Hello prompt generation:")
    print(f"  Input: {tokens[0]}")
    print(f"  Generated: {generated[:3]}...")

if __name__ == "__main__":
    main()