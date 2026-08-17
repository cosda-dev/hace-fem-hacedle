#[cfg(feature = "std")]
use std::sync::Arc;

pub struct GgufMmap {
    pub ptr: *mut u8,
    pub size: usize,
}

impl GgufMmap {
    pub fn new(ptr: *mut u8, size: usize) -> Self {
        Self { ptr, size }
    }

    #[cfg(feature = "std")]
    pub fn map_file(path: &str) -> Result<Self, &'static str> {
        Ok(Self { ptr: std::ptr::null_mut(), size: 0 })
    }
}
