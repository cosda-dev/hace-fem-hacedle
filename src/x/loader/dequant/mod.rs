mod q4_k;
mod q6_k;

pub use q4_k::dequant_q4_k;
pub use q6_k::dequant_q6_k;

pub enum QuantType {
    F32,
    F16,
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
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
            QuantType::Q8_0 => QuantSpec { block_size: 32, bytes_per_block: 32 },
            QuantType::F16 => QuantSpec { block_size: 1, bytes_per_block: 2 },
            QuantType::F32 => QuantSpec { block_size: 1, bytes_per_block: 4 },
        }
    }
}