use crate::core::{TensorI16, TensorI8};

pub const SKB_TOTAL_BYTES: usize = 4096;
pub const SKB_HEADER_BYTES: usize = 64;

pub const SKB_TENSOR_OFFSET: usize = SKB_HEADER_BYTES;
pub const SKB_TENSOR_BYTES: usize = 2048;

pub const SKB_LOGIC_OFFSET: usize = SKB_TENSOR_OFFSET + SKB_TENSOR_BYTES;
pub const SKB_LOGIC_I16_LEN: usize = 512; // 512 * 2 = 1024 bytes
pub const SKB_LOGIC_BYTES: usize = SKB_LOGIC_I16_LEN * 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkbViewError {
    InvalidSize,
    Misaligned,
}

pub struct SkbSliceView<'a> {
    pub tensor_i8: &'a [i8],
    pub logic_i16: &'a [i16],
}

#[cfg(feature = "std")]
pub struct SkbMmap {
    mmap: memmap2::Mmap,
}

pub fn view_from_skb(bytes: &[u8]) -> Result<SkbSliceView<'_>, SkbViewError> {
    if bytes.len() < SKB_LOGIC_OFFSET + SKB_LOGIC_BYTES {
        return Err(SkbViewError::InvalidSize);
    }
    if SKB_LOGIC_OFFSET % core::mem::align_of::<i16>() != 0 {
        return Err(SkbViewError::Misaligned);
    }

    let tensor = &bytes[SKB_TENSOR_OFFSET..SKB_TENSOR_OFFSET + SKB_TENSOR_BYTES];
    let tensor_i8 = unsafe { core::slice::from_raw_parts(tensor.as_ptr() as *const i8, tensor.len()) };

    let logic = &bytes[SKB_LOGIC_OFFSET..SKB_LOGIC_OFFSET + SKB_LOGIC_BYTES];
    let logic_i16 = unsafe {
        core::slice::from_raw_parts(
            logic.as_ptr() as *const i16,
            SKB_LOGIC_I16_LEN,
        )
    };

    Ok(SkbSliceView { tensor_i8, logic_i16 })
}

#[cfg(feature = "std")]
pub fn mmap_skb(path: &std::path::Path) -> Result<SkbMmap, SkbViewError> {
    use std::fs::File;
    let file = File::open(path).map_err(|_| SkbViewError::InvalidSize)?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
        .map_err(|_| SkbViewError::InvalidSize)?;
    Ok(SkbMmap { mmap })
}

#[cfg(feature = "std")]
impl SkbMmap {
    pub fn view(&self) -> Result<SkbSliceView<'_>, SkbViewError> {
        view_from_skb(self.mmap.as_ref())
    }

    pub fn bytes(&self) -> &[u8] {
        self.mmap.as_ref()
    }
}

pub fn tensor_from_skb(bytes: &[u8]) -> Result<TensorI8, SkbViewError> {
    let view = view_from_skb(bytes)?;
    Ok(TensorI8::new(view.tensor_i8.to_vec()))
}

pub fn logic_from_skb(bytes: &[u8]) -> Result<TensorI16, SkbViewError> {
    let view = view_from_skb(bytes)?;
    Ok(TensorI16::new(view.logic_i16.to_vec()))
}
