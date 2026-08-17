use alloc::string::String;

use super::PayloadView;

#[derive(Debug)]
pub enum ExecError {
    ExecutorNotFound(String),
    DispatchFailed(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmoVendorId {
    Llama,
    Onnx,
    Mlx,
    Vllm,
    Candle,
}

#[derive(Debug, Clone)]
pub struct ExecutionHandle {
    pub executor_id: String,
    pub session_id: u64,
    pub is_active: bool,
}

pub trait SkbDispatchNep {
    fn select_executor(&self, payload: &PayloadView) -> AmoVendorId;

    fn dispatch(&self, payload: &PayloadView) -> Result<ExecutionHandle, ExecError>;
}
