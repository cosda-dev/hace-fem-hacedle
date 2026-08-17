// KV Cache Parity Test - Verify cache operations
// Run: cargo test --test kv_parity --features std -- --nocapture

#[test]
fn test_kv_cache_append() {
    use hace_fem_hacedle::core::ops::block::KvCache;
    
    let n_heads = 28;
    let head_dim = 128;
    let max_seq_len = 100;
    
    let mut cache = KvCache::new(max_seq_len, n_heads, head_dim);
    
    for pos in 0..10 {
        let k = vec![pos as f32; n_heads * head_dim];
        let v = vec![(pos * 2) as f32; n_heads * head_dim];
        cache.append(&k, &v, pos);
    }
    
    assert_eq!(cache.current_len(), 10);
    
    for head in 0..n_heads {
        for idx in 0..10 {
            let k_val = cache.get_cached_k(head, idx, head_dim);
            let v_val = cache.get_cached_v(head, idx, head_dim);
            
            assert!((k_val - idx as f32).abs() < 1e-5, "K cache mismatch at pos {}", idx);
            assert!((v_val - idx as f32 * 2.0).abs() < 1e-5, "V cache mismatch at pos {}", idx);
        }
    }
}

#[test]
fn test_kv_cache_bounds() {
    use hace_fem_hacedle::core::ops::block::KvCache;
    
    let n_heads = 1;
    let head_dim = 4;
    let max_seq_len = 10;
    
    let mut cache = KvCache::new(max_seq_len, n_heads, head_dim);
    
    let k = vec![1.0, 2.0, 3.0, 4.0];
    let v = vec![10.0, 20.0, 30.0, 40.0];
    
    cache.append(&k, &v, 0);
    
    assert!((cache.get_cached_k(0, 0, 4) - 1.0).abs() < 1e-5);
    assert!((cache.get_cached_v(0, 0, 4) - 10.0).abs() < 1e-5);
    
    cache.append(&k, &v, 1);
    
    assert_eq!(cache.current_len(), 2);
    assert!((cache.get_cached_k(0, 1, 4) - 1.0).abs() < 1e-5);
    assert!((cache.get_cached_v(0, 1, 4) - 10.0).abs() < 1e-5);
}

#[test]
fn test_kv_cache_overflow_protection() {
    use hace_fem_hacedle::core::ops::block::KvCache;
    
    let n_heads = 1;
    let head_dim = 4;
    let max_seq_len = 5;
    
    let mut cache = KvCache::new(max_seq_len, n_heads, head_dim);
    
    for pos in 0..100 {
        let k = vec![1.0; n_heads * head_dim];
        let v = vec![2.0; n_heads * head_dim];
        cache.append(&k, &v, pos);
    }
    
    assert!(cache.current_len() <= max_seq_len);
}