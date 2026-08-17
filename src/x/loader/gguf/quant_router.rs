use alloc::string::String;
use alloc::vec::Vec;
use alloc::vec;
use super::header::QuantizationType;

pub struct QuantRouter;

impl QuantRouter {
    pub fn route(quant: QuantizationType) -> Vec<&'static str> {
        match quant {
            QuantizationType::Q4K_M => vec!["candle", "llama"],
            QuantizationType::Q5K_M => vec!["candle", "llama", "mlx"],
            QuantizationType::Q6K => vec!["llama", "mlx"],
            QuantizationType::IQ4XS => vec!["llama", "onnx"],
            QuantizationType::IQ3M => vec!["llama"],
            QuantizationType::Q4K_S => vec!["candle", "llama"],
            QuantizationType::Q5K_S => vec!["candle", "llama"],
            QuantizationType::F16 => vec!["mlx", "vllm"],
            QuantizationType::F32 => vec!["vllm"],
            _ => vec!["candle"],
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantizationRoute {
    pub quant_type: QuantizationType,
    pub providers: Vec<String>,
}
