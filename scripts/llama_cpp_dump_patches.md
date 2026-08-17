# llama.cpp Activation Dump Patches
# Add these patches to llama.cpp for A3-10 parity testing

## Patch 1: Add dump_dir global variable
# In llama.cpp or common.cpp:
```cpp
static std::string g_dump_dir = "reference/";
static bool g_dump_activations = false;
```

## Patch 2: Add dump_tensor helper function
```cpp
void dump_tensor(const char* name, const float* data, size_t size) {
    if (!g_dump_activations) return;
    
    std::string path = g_dump_dir + name + ".bin";
    FILE* f = fopen(path.c_str(), "wb");
    if (f) {
        fwrite(data, sizeof(float), size, f);
        fclose(f);
    }
}
```

## Patch 3: Dump Block0 activations in llama_build_forward_plan
# In llama.cpp, inside the block loop:
```cpp
// After RMSNorm
dump_tensor("block0_norm", inp, n_embd);

// After QKV projection
dump_tensor("block0_q", q, n_embd);
dump_tensor("block0_k", k, n_embd);
dump_tensor("block0_v", v, n_embd);

// After RoPE
dump_tensor("block0_q_rope", q, n_embd);
dump_tensor("block0_k_rope", k, n_embd);

// After Attention scores
for (int i = 0; i < n_head; i++) {
    dump_tensor(("block0_score_h" + std::to_string(i)).c_str(), 
               attention_scores + i*seq_len, seq_len);
}

// After Softmax
dump_tensor("block0_attn_prob", attention_probs, n_head*seq_len);

// After Attention output
dump_tensor("block0_attn_out", cur, n_embd);

// After FFN gate/up/down
dump_tensor("block0_ffn_gate", ffn_gate_out, n_ff);
dump_tensor("block0_ffn_up", ffn_up_out, n_ff);
dump_tensor("block0_ffn_silu", ffn_silu_out, n_ff);
dump_tensor("block0_ffn_down", ffn_down_out, n_embd);

// After residual
dump_tensor("block0_out", inp, n_embd);
```

## Patch 4: Dump final logits
# In llama_decode or llama_eval:
```cpp
dump_tensor("final_logits", logits, n_vocab);
```

## Patch 5: Command line flag
# Add to llama-cli args:
```cpp
if (strcmp(argv[i], "--dump-activations") == 0) {
    g_dump_activations = true;
    g_dump_dir = argv[++i];
}
```

## Usage:
```bash
./llama-cli --model Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
           --dump-activations dump_output/ \
           -p "Hello" -n 1
```

## Expected Output Files:
```
dump_output/
├── block0_norm.bin
├── block0_q.bin
├── block0_k.bin
├── block0_v.bin
├── block0_q_rope.bin
├── block0_k_rope.bin
├── block0_attn_prob.bin
├── block0_attn_out.bin
├── block0_ffn_gate.bin
├── block0_ffn_up.bin
├── block0_ffn_silu.bin
├── block0_ffn_down.bin
├── block0_out.bin
└── final_logits.bin
```