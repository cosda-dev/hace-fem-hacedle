// Q4_K_M Dequantization - Based on llama.cpp format
// Block: 256 elements, 192 bytes total

use crate::alloc_exports::*;

/// Convert f16 (u16) to f32
fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1F;
    let frac = h & 0x3FF;
    
    if exp == 0 {
        return 0.0;
    }
    
    let f32_exp = (exp as i32) - 15 + 127;
    let f32_frac = frac as f32 / 1024.0;
    let result = (1.0 + f32_frac) * 2.0_f32.powi(f32_exp);
    if sign == 1 { -result } else { result }
}

/// Dequantize Q4_K tensor to f32
/// Format: Per 256-element block:
///   - 16 f16 scales (32 bytes)
///   - 16 f16 mins (32 bytes)
///   - 256 nibbles packed as u4 (128 bytes)
///   Total: 192 bytes per 256 elements
pub fn dequant_q4_k(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 192; // 32 + 32 + 128 bytes
    
    let blocks = data.len() / BYTES_PER_BLOCK;
    let elements = output.len().min(blocks * BLOCK_SIZE);
    
    for block_idx in 0..blocks {
        let block_start = block_idx * BYTES_PER_BLOCK;
        if block_start + BYTES_PER_BLOCK > data.len() {
            break;
        }
        let block = &data[block_start..block_start + BYTES_PER_BLOCK];
        
        // Extract scales (first 16 f16 values, 32 bytes)
        let mut scales = [0.0f32; 16];
        for i in 0..16 {
            let offset = i * 2;
            let h = u16::from_le_bytes([block[offset], block[offset + 1]]);
            scales[i] = f16_to_f32(h);
        }
        
        // Extract mins (next 16 f16 values, 32 bytes)
        let mut mins = [0.0f32; 16];
        for i in 0..16 {
            let offset = 32 + i * 2;
            let h = u16::from_le_bytes([block[offset], block[offset + 1]]);
            mins[i] = f16_to_f32(h);
        }
        
        // Extract quantized values (remaining 128 bytes, packed nibbles)
        // Each byte contains 2 nibbles (4-bit values)
        let quants_start = 64;
        let quants = &block[quants_start..quants_start + 128];
        
        for j in 0..BLOCK_SIZE {
            let idx = block_idx * BLOCK_SIZE + j;
            if idx >= elements {
                break;
            }
            
            // Determine which scale/min to use (16 groups of 16)
            let g = j / 16;
            let j_in_group = j % 16;
            
            // Get nibble value (0-15)
            let nibble_offset = (j / 2) as usize;
            let q = if j % 2 == 0 {
                quants[nibble_offset] & 0x0F
            } else {
                quants[nibble_offset] >> 4
            };
            
            // Calculate value: scales[g] * (q as f32) + mins[g]
            output[idx] = scales[g] * (q as f32) + mins[g];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_f16_zero() {
        let h: u16 = 0;
        assert_eq!(f16_to_f32(h), 0.0);
    }
    
    #[test]
    fn test_dequant_empty() {
        let data: [u8; 192] = [0; 192]; // One block of zeros
        let mut output = [0.0f32; 256];
        dequant_q4_k(&data, &mut output);
        // All zeros in -> all zeros out
        assert_eq!(output[0], 0.0);
    }
}