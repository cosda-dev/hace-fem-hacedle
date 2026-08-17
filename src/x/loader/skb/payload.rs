use alloc::string::String;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadType {
    Gguf,
    Onnx,
    Mlx,
    Custom,
}

#[derive(Debug, Clone)]
pub struct PayloadView {
    pub payload_type: PayloadType,
    pub location: Location,
    pub offset: u64,
    pub size: u64,
    pub executor_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Location {
    Internal,
    External,
}

impl PayloadView {
    pub fn select_executor(&self) -> Option<String> {
        match self.payload_type {
            PayloadType::Gguf => Some(String::from("x/llama")),
            PayloadType::Onnx => Some(String::from("x/onnx")),
            PayloadType::Mlx => Some(String::from("x/mlx")),
            PayloadType::Custom => self.executor_id.clone(),
        }
    }
}
