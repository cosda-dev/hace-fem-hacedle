use alloc::string::String;
use alloc::vec::Vec;

pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    Q4K_M,
    Q5K_M,
    Q6K,
    Q3K_M,
    Q4K_S,
    Q5K_S,
    IQ2K,
    IQ3M,
    IQ4XS,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    F16,
    F32,
    Unknown,
}

impl QuantizationType {
    pub fn as_str(&self) -> &'static str {
        match self {
            QuantizationType::Q4K_M => "q4_k_m",
            QuantizationType::Q5K_M => "q5_k_m",
            QuantizationType::Q6K => "q6_k",
            QuantizationType::Q3K_M => "q3_k_m",
            QuantizationType::Q4K_S => "q4_k_s",
            QuantizationType::Q5K_S => "q5_k_s",
            QuantizationType::IQ2K => "iq2_k",
            QuantizationType::IQ3M => "iq3_m",
            QuantizationType::IQ4XS => "iq4_xs",
            QuantizationType::Q4_0 => "q4_0",
            QuantizationType::Q4_1 => "q4_1",
            QuantizationType::Q5_0 => "q5_0",
            QuantizationType::Q5_1 => "q5_1",
            QuantizationType::Q8_0 => "q8_0",
            QuantizationType::F16 => "f16",
            QuantizationType::F32 => "f32",
            QuantizationType::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub magic: [u8; 4],
    pub version: [u8; 4],
    pub tensor_count: u64,
    pub kv_count: u64,
    pub metadata: Vec<GgufMetadata>,
}

#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub key: String,
    pub value: GgufValue,
}

#[derive(Debug, Clone)]
pub enum GgufValue {
    String(String),
    I32(i32),
    I64(i64),
    F32(f32),
    I8(u8),
    Bool(bool),
    ArrayString(Vec<String>),
    ArrayI32(Vec<i32>),
    ArrayI64(Vec<i64>),
    ArrayF32(Vec<f32>),
}

impl GgufHeader {
    pub fn is_valid(&self) -> bool {
        self.magic == GGUF_MAGIC
    }
}
