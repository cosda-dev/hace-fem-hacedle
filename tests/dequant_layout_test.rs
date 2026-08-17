// P2: Q5_0/Q6_K Block Layout Verification
// Verify bytes_per_block matches llama.cpp spec

#[test]
fn test_q5_0_layout() {
    use crate::quant_view::{QuantType, QuantSpec};
    
    let spec = QuantType::Q5_0.spec();
    assert_eq!(spec.block_size, 32, "Q5_0 block size should be 32");
    assert_eq!(spec.bytes_per_block, 22, "Q5_0 bytes per block should be 22 (2+4+16)");
}

#[test]
fn test_q6_k_layout() {
    use crate::quant_view::{QuantType, QuantSpec};
    
    let spec = QuantType::Q6K.spec();
    assert_eq!(spec.block_size, 256, "Q6_K block size should be 256");
    assert_eq!(spec.bytes_per_block, 210, "Q6_K bytes per block should be 210");
}

#[test]
fn test_q8_0_layout() {
    use crate::quant_view::{QuantType, QuantSpec};
    
    let spec = QuantType::Q8_0.spec();
    assert_eq!(spec.block_size, 32, "Q8_0 block size should be 32");
    assert_eq!(spec.bytes_per_block, 34, "Q8_0 bytes per block should be 34 (2+32)");
}

#[test]
fn test_q4_k_layout() {
    use crate::quant_view::{QuantType, QuantSpec};
    
    let spec = QuantType::Q4K.spec();
    assert_eq!(spec.block_size, 256, "Q4_K block size should be 256");
    assert_eq!(spec.bytes_per_block, 144, "Q4_K bytes per block should be 144");
}