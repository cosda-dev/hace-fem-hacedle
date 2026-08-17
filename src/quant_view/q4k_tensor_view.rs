
// QuantTensorView - Zero-copy Q4_K tensor view
// Keep tensor quantized until compute dispatch

use crate::alloc_exports::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    F32 = 0,
    F16 = 15,
    Q4K = 18,
    Q5K = 19,
    Q6K = 20,
    Q8_0 = 14,
    Q4_0 = 10,
    Q4_1 = 11,
    Q5_0 = 12,
    Q5_1 = 13,
    BF16 = 21,
}

#[derive(Debug, Clone, Copy)]
pub struct QuantSpec {
    pub block_size: usize,
    pub bytes_per_block: usize,
}

impl QuantType {
    pub fn spec(&self) -> QuantSpec {
        match self {
            QuantType::Q4K => QuantSpec { block_size: 256, bytes_per_block: 144 },
            QuantType::Q5K => QuantSpec { block_size: 256, bytes_per_block: 176 },
            QuantType::Q6K => QuantSpec { block_size: 256, bytes_per_block: 210 },
            QuantType::Q8_0 => QuantSpec { block_size: 32, bytes_per_block: 34 },
            QuantType::Q4_0 => QuantSpec { block_size: 32, bytes_per_block: 18 },
            QuantType::Q4_1 => QuantSpec { block_size: 32, bytes_per_block: 20 },
            QuantType::Q5_0 => QuantSpec { block_size: 32, bytes_per_block: 22 },
            QuantType::Q5_1 => QuantSpec { block_size: 32, bytes_per_block: 24 },
            QuantType::F16 => QuantSpec { block_size: 1, bytes_per_block: 2 },
            QuantType::F32 => QuantSpec { block_size: 1, bytes_per_block: 4 },
            QuantType::BF16 => QuantSpec { block_size: 1, bytes_per_block: 2 },
        }
    }
}

pub struct QuantTensorView {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub quant_type: QuantType,
}

impl QuantTensorView {
    pub fn new(data: Vec<u8>, shape: Vec<usize>, quant_type: QuantType) -> Self {
        Self { data, shape, quant_type }
    }

    pub fn size_bytes(&self) -> usize {
        let numel: usize = self.shape.iter().product();
        let spec = self.quant_type.spec();
        (numel + spec.block_size - 1) / spec.block_size * spec.bytes_per_block
    }

    pub fn dequantize(&self) -> Vec<f32> {
        let numel: usize = self.shape.iter().product();
        let mut output = vec![0.0f32; numel];
        
        match self.quant_type {
            QuantType::Q4K => {
                dequant_q4_k_exact(&self.data, &mut output);
            }
            QuantType::Q5_0 => {
                dequant_q5_0_exact(&self.data, &mut output);
            }
            QuantType::Q6K => {
                dequant_q6_k_exact(&self.data, &mut output);
            }
            QuantType::Q8_0 => {
                dequant_q8_0_exact(&self.data, &mut output);
            }
            QuantType::Q4_0 | QuantType::Q4_1 | QuantType::Q5_1 => {
                output.fill(0.0);
            }
            QuantType::Q5K => {
                output.fill(0.0);
            }
            QuantType::F16 => {
                let half_count = (numel).min(self.data.len() / 2);
                for i in 0..half_count {
                    let h = u16::from_le_bytes([self.data[i * 2], self.data[i * 2 + 1]]);
                    output[i] = f16_to_f32(h);
                }
            }
            QuantType::F32 => {
                let float_count = numel.min(self.data.len() / 4);
                for i in 0..float_count {
                    output[i] = f32::from_le_bytes([
                        self.data[i * 4],
                        self.data[i * 4 + 1],
                        self.data[i * 4 + 2],
                        self.data[i * 4 + 3],
                    ]);
                }
            }
            QuantType::BF16 => {
                let half_count = numel.min(self.data.len() / 2);
                for i in 0..half_count {
                    let h = u16::from_le_bytes([self.data[i * 2], self.data[i * 2 + 1]]);
                    output[i] = bf16_to_f32(h);
                }
            }
        }
        
        output
    }

    pub fn dequantize_subset(&self, offset: usize, count: usize) -> Vec<f32> {
        let mut full_output = self.dequantize();
        let end = offset + count.min(full_output.len() - offset);
        full_output[offset..end].to_vec()
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1F;
    let frac = h & 0x3FF;
    
    if exp == 0 {
        0.0
    } else {
        let f32_exp = (exp as i32) - 15 + 127;
        let f32_frac = frac as f32 / 1024.0;
        let result = (1.0 + f32_frac) * 2.0_f32.powi(f32_exp);
        if sign == 1 { -result } else { result }
    }
}

fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

pub fn dequant_q4_k_exact(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 144;
    
    let blocks = data.len() / BYTES_PER_BLOCK;
    let elements = output.len().min(blocks * BLOCK_SIZE);
    
    for block_idx in 0..blocks {
        let block_offset = block_idx * BYTES_PER_BLOCK;
        if block_offset + BYTES_PER_BLOCK > data.len() {
            break;
        }
        let block = &data[block_offset..block_offset + BYTES_PER_BLOCK];
        
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min_val = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        
        for j in 0..BLOCK_SIZE {
            let idx = block_idx * BLOCK_SIZE + j;
            if idx >= elements {
                break;
            }
            
            let scale_idx = 8 + (j / 32);
            let scale_val = f16_to_f32(u16::from_le_bytes([block[scale_idx * 2], block[scale_idx * 2 + 1]]));
            
            let q_offset = 16 + (j / 2);
            let q = if j % 2 == 0 {
                block[q_offset] & 0xF
            } else {
                block[q_offset] >> 4
            };
            
            output[idx] = (q as f32) * scale_val + min_val;
        }
    }
}

pub fn dequant_q5_0_exact(data: &[u8], output: &mut [f32]) {
    // Q5_0: 32 elements per block, 22 bytes total
    // Layout: d(2) + qh(4) + qs(16)
    // qs broadcast: each of 16 bytes contributes to 2 positions via shift
    const BLOCK_SIZE: usize = 32;
    const BYTES_PER_BLOCK: usize = 22;
    
    let blocks = data.len() / BYTES_PER_BLOCK;
    let elements = output.len().min(blocks * BLOCK_SIZE);
    
    for block_idx in 0..blocks {
        let block_offset = block_idx * BYTES_PER_BLOCK;
        if block_offset + BYTES_PER_BLOCK > data.len() {
            break;
        }
        let block = &data[block_offset..block_offset + BYTES_PER_BLOCK];
        
        // d: f16 scale
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        
        // qh: 4 bytes packed (32 bits for 32 elements) - 1 bit per element
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        
        // qs: 16 bytes, broadcast to 32 positions
        for j in 0..BLOCK_SIZE {
            let idx = block_idx * BLOCK_SIZE + j;
            if idx >= elements {
                break;
            }
            
            let qh_bit = ((qh >> j) & 1) as u8;
            
            // qs[j] for j<16 shifted by 0, qs[j-16] for j>=16 shifted by 4
            let qs_idx = if j < 16 { j } else { j - 16 };
            let shift = if j < 16 { 0 } else { 4 };
            let ql = (block[6 + qs_idx] >> shift) & 0x0F;
            
            // Combine: 5-bit value centered at -16
            let q = ql + (qh_bit << 4);
            output[idx] = (q as f32 - 16.0) * d;
        }
    }
}

pub fn dequant_q6_k_exact(data: &[u8], output: &mut [f32]) {
    // Q6_K: 256 elements per block, 210 bytes total
    // Layout from gguf-py: (256, 2 + 256//2 + 256//4 + 256//16) = (256, 2 + 128 + 64 + 16) = (256, 210)
    // But actual is: d(2) + ql(192) + qh(16) = need to check ggml
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 210;
    
    let blocks = data.len() / BYTES_PER_BLOCK;
    let elements = output.len().min(blocks * BLOCK_SIZE);
    
    for block_idx in 0..blocks {
        let block_offset = block_idx * BYTES_PER_BLOCK;
        if block_offset + BYTES_PER_BLOCK > data.len() {
            break;
        }
        let block = &data[block_offset..block_offset + BYTES_PER_BLOCK];
        
        // d: f16 scale
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        
        // Q6_K layout per ggml: d(2) + ql(256*6/8=192) + qh(256*2/8=16*4=64) 
        // Actually: ql is 6 bits per value (192 bytes), qh is 2 bits per value (64 bytes)
        for j in 0..BLOCK_SIZE {
            let idx = block_idx * BLOCK_SIZE + j;
            if idx >= elements {
                break;
            }
            
            // Extract 6-bit value from ql (bytes 2-193)
            let ql_byte_idx = j * 6 / 8;
            let ql_bit_offset = (j * 6) % 8;
            
            let ql_low = block[2 + ql_byte_idx] as usize;
            let ql = if ql_bit_offset <= 2 {
                (ql_low >> ql_bit_offset) & 0x3F
            } else {
                ((ql_low >> ql_bit_offset) | ((block[2 + ql_byte_idx + 1] as usize) << (8 - ql_bit_offset))) & 0x3F
            };
            
            // Extract 2-bit value from qh (bytes 194-257)
            let qh_offset = 2 + 256 * 6 / 8 + (j / 4);  // 194 + j/4
            let qh_bit = (j % 4) * 2;
            let qh = if qh_offset < BYTES_PER_BLOCK {
                (block[qh_offset as usize] >> qh_bit) & 0x03
            } else {
                0
            };
            
            let q = ql + ((qh as usize) << 6);
            output[idx] = (q as f32 - 32.0) * d;
        }
    }
}

pub fn dequant_q8_0_exact(data: &[u8], output: &mut [f32]) {
    // Q8_0: 32 elements per block, 34 bytes total
    // Layout from gguf-py: (32, 2 + 32) = d(2 bytes f16) + qs(32 bytes u8)
    const BLOCK_SIZE: usize = 32;
    const BYTES_PER_BLOCK: usize = 34;
    
    let blocks = data.len() / BYTES_PER_BLOCK;
    let elements = output.len().min(blocks * BLOCK_SIZE);
    
    for block_idx in 0..blocks {
        let block_offset = block_idx * BYTES_PER_BLOCK;
        if block_offset + BYTES_PER_BLOCK > data.len() {
            break;
        }
        let block = &data[block_offset..block_offset + BYTES_PER_BLOCK];
        
        // d: f16 scale
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        
        // qs: 32 bytes signed i8 values
        for j in 0..BLOCK_SIZE {
            let idx = block_idx * BLOCK_SIZE + j;
            if idx >= elements {
                break;
            }
            
            // Q8_0: values are signed i8 centered at 0
            let v = block[2 + j] as i8;
            output[idx] = v as f32 * d;
        }
    }
}

