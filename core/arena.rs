#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

pub struct Arena {
    buffer: Vec<u8>,
    offset: usize,
}

impl Arena {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: vec![0u8; capacity],
            offset: 0,
        }
    }

    pub fn alloc_bytes(&mut self, size: usize) -> Option<&mut [u8]> {
        if self.offset.saturating_add(size) > self.buffer.len() {
            return None;
        }
        let ptr = unsafe { self.buffer.as_mut_ptr().add(self.offset) };
        self.offset += size;
        Some(unsafe { core::slice::from_raw_parts_mut(ptr, size) })
    }

    pub fn reset(&mut self) {
        self.offset = 0;
    }
}
