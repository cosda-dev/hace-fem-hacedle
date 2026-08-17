// GGUF Tensor Index Parser - CRD Directive P0-2 Phase 2
// Parses tensor metadata from GGUF format

use alloc::string::String;
use alloc::vec::Vec;
use alloc::vec;

use super::header::{QuantizationType, GgufHeader};

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub ggml_type: u32,
    pub n_elements: usize,
}

pub struct GgufTensorIndex {
    pub tensors: Vec<TensorInfo>,
}

impl GgufTensorIndex {
    pub fn parse(mmap: &[u8], header: &GgufHeader) -> Result<Self, &'static str> {
        let mut tensors = Vec::new();
        
        // Offset after header
        // GGUF layout: magic(4) + version(4) + tensor_count(8) + kv_count(8) + metadata + tensor_index
        let mut offset = 24usize; // After magic+version+tensor_count+kv_count
        
        // Skip metadata key-value pairs
        for _ in 0..header.kv_count {
            if offset + 8 > mmap.len() {
                break;
            }
            let key_len = u64::from_le_bytes([
                mmap[offset], mmap[offset+1], mmap[offset+2], mmap[offset+3],
                mmap[offset+4], mmap[offset+5], mmap[offset+6], mmap[offset+7]
            ]) as usize;
            offset += 8;
            
            if offset + key_len > mmap.len() {
                break;
            }
            offset += key_len; // Skip key string
            
            // Skip value based on type byte
            if offset >= mmap.len() {
                break;
            }
            let type_byte = mmap[offset];
            offset += 1;
            
            match type_byte {
                0 => {} // UINT8 (1 byte)
                1 => { if offset + 1 > mmap.len() { break; } offset += 1; } // INT8
                2 => { if offset + 2 > mmap.len() { break; } offset += 2; } // UINT16
                3 => { if offset + 2 > mmap.len() { break; } offset += 2; } // INT16
                4 => { if offset + 4 > mmap.len() { break; } offset += 4; } // UINT32
                5 => { if offset + 4 > mmap.len() { break; } offset += 4; } // INT32
                6 => { if offset + 4 > mmap.len() { break; } offset += 4; } // FLOAT32
                7 => { if offset + 1 > mmap.len() { break; } offset += 1; } // BOOL
                8 => { if offset + 8 > mmap.len() { break; } offset += 8; } // UINT64
                9 => { if offset + 8 > mmap.len() { break; } offset += 8; } // INT64
                10 => { if offset + 8 > mmap.len() { break; } offset += 8; } // FLOAT64
                11 => { // STRING
                    if offset + 8 > mmap.len() { break; }
                    let s_len = u64::from_le_bytes([
                        mmap[offset], mmap[offset+1], mmap[offset+2], mmap[offset+3],
                        mmap[offset+4], mmap[offset+5], mmap[offset+6], mmap[offset+7]
                    ]) as usize;
                    offset += 8;
                    if offset + s_len > mmap.len() { break; }
                    offset += s_len;
                }
                12 => { // ARRAY
                    if offset + 1 > mmap.len() { break; }
                    offset += 1; // element type
                    if offset + 8 > mmap.len() { break; }
                    let arr_len = u64::from_le_bytes([
                        mmap[offset], mmap[offset+1], mmap[offset+2], mmap[offset+3],
                        mmap[offset+4], mmap[offset+5], mmap[offset+6], mmap[offset+7]
                    ]) as usize;
                    offset += 8;
                    // We need to skip array elements based on element type, but for simplicity just skip
                    // Each element is up to 8 bytes max (FLOAT64/UINT64)
                    if offset + arr_len * 8 <= mmap.len() {
                        offset += arr_len * 8; // Over-allocate to skip safely
                    } else if offset + arr_len * 4 <= mmap.len() {
                        offset += arr_len * 4;
                    }
                }
                _ => { break; }
            }
        }
        
        // Parse tensor index
        for _ in 0..header.tensor_count {
            // Tensor name length (u64)
            if offset + 8 > mmap.len() {
                break;
            }
            let name_len = u64::from_le_bytes([
                mmap[offset], mmap[offset+1], mmap[offset+2], mmap[offset+3],
                mmap[offset+4], mmap[offset+5], mmap[offset+6], mmap[offset+7]
            ]) as usize;
            offset += 8;
            
            // Tensor name
            if offset + name_len > mmap.len() {
                break;
            }
            let name = String::from_utf8_lossy(&mmap[offset..offset+name_len]).to_string();
            offset += name_len;
            
            // Dtype (u32) - store as ggml_type directly
            if offset + 4 > mmap.len() {
                break;
            }
            let ggml_type = u32::from_le_bytes([
                mmap[offset], mmap[offset+1], mmap[offset+2], mmap[offset+3]
            ]);
            offset += 4;
            
            // N_dims (u32)
            if offset + 4 > mmap.len() {
                break;
            }
            let n_dims = u32::from_le_bytes([
                mmap[offset], mmap[offset+1], mmap[offset+2], mmap[offset+3]
            ]) as usize;
            offset += 4;
            
            // Shape (n_dims * u64)
            let mut shape = Vec::new();
            for _ in 0..n_dims {
                if offset + 8 > mmap.len() {
                    break;
                }
                let dim = u64::from_le_bytes([
                    mmap[offset], mmap[offset+1], mmap[offset+2], mmap[offset+3],
                    mmap[offset+4], mmap[offset+5], mmap[offset+6], mmap[offset+7]
                ]) as usize;
                shape.push(dim);
                offset += 8;
            }
            
            // Tensor offset (u64)
            if offset + 8 > mmap.len() {
                break;
            }
            let tensor_offset = u64::from_le_bytes([
                mmap[offset], mmap[offset+1], mmap[offset+2], mmap[offset+3],
                mmap[offset+4], mmap[offset+5], mmap[offset+6], mmap[offset+7]
            ]);
            offset += 8;
            
            let n_elements: usize = shape.iter().product();
            
            tensors.push(TensorInfo {
                name,
                shape,
                offset: tensor_offset,
                ggml_type,
                n_elements,
            });
        }
        
        Ok(Self { tensors })
    }
}

impl Default for GgufTensorIndex {
    fn default() -> Self {
        Self { tensors: Vec::new() }
    }
}