import gguf
import sys

model_path = r"D:\host\llama-models\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"

with open(model_path, 'rb') as f:
    reader = gguf.GGUFReader(f)
    
    print("=== GGUF Metadata Golden Test ===")
    print()
    
    arch = reader.get_attr('general.architecture')
    block_count = reader.get_attr('qwen2.block_count')
    head_count = reader.get_attr('qwen2.attention.head_count')
    head_count_kv = reader.get_attr('qwen2.attention.head_count_kv')
    embed_len = reader.get_attr('qwen2.embedding_length')
    context_len = reader.get_attr('qwen2.context_length')
    rope_theta = reader.get_attr('qwen2.rope.freq_base')
    
    print(f"arch = {arch}")
    print(f"block_count = {block_count}")
    print(f"head_count = {head_count}")
    print(f"head_count_kv = {head_count_kv}")
    print(f"embedding_length = {embed_len}")
    print(f"context_length = {context_len}")
    print(f"rope_theta = {rope_theta}")
    print()
    
    passed = 0
    failures = []
    
    if arch == 'qwen2':
        print('✅ general.architecture = qwen2')
        passed += 1
    else:
        print(f'❌ general.architecture = {arch} (expected: qwen2)')
        failures.append('general.architecture')
    
    if block_count == 28:
        print('✅ qwen2.block_count = 28')
        passed += 1
    else:
        print(f'❌ qwen2.block_count = {block_count} (expected: 28)')
        failures.append('block_count')
    
    if head_count == 12:
        print('✅ qwen2.attention.head_count = 12')
        passed += 1
    else:
        print(f'❌ qwen2.attention.head_count = {head_count} (expected: 12)')
        failures.append('head_count')
    
    print()
    print(f"=== Summary: {passed}/3 passed ===")
    
    if failures:
        print(f"Failures: {failures}")
        sys.exit(1)
    else:
        print("✅ V1.1 TEST PASSED")