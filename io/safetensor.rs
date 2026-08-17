#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::core::TensorI8;

/// Minimal safetensor loader stub: treat raw bytes as i8 tensor payload.
pub fn load_safetensor_i8(bytes: &[u8]) -> TensorI8 {
    let mut data = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        data.push(bytes[i] as i8);
        i += 1;
    }
    TensorI8::new(data)
}
