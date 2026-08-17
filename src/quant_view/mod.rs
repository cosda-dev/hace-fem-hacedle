mod q4k_tensor_view;
mod dequant_dispatcher;

pub use q4k_tensor_view::{QuantTensorView, QuantType, dequant_q4_k_exact, dequant_q5_0_exact, dequant_q6_k_exact, dequant_q8_0_exact};
#[cfg(feature = "std")]
pub use dequant_dispatcher::{DequantDispatcher, ComputeBackend, NativeBackend};