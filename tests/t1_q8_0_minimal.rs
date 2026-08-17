// T1: Q8_0 Bit Exact - Minimal standalone test
// Run only if parity_test/q8_0/*.raw exists

use std::fs;
use std::path::Path;

#[test]
fn test_q8_0_data_exists() {
    let raw_path = Path::new("parity_test/q8_0/blk0_attn_v_q8_0_first_block.raw");
    let ref_path = Path::new("parity_test/q8_0/blk0_attn_v_q8_0_first_block.bin");
    
    assert!(raw_path.exists(), "Run t1_q8_0_parity_extract.py first");
    assert!(ref_path.exists(), "Reference file missing");
    
    let raw = fs::read(raw_path).unwrap();
    assert_eq!(raw.len(), 34, "Q8_0 block should be 34 bytes");
    
    let ref_data = fs::read(ref_path).unwrap();
    assert_eq!(ref_data.len(), 128, "32 f32 values = 128 bytes");
}