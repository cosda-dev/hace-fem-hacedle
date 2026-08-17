
use core::{mem, slice};

use crate::core::tensor::{TensorError, TensorView};
use crate::rr_bind::blob_source::{BlobError, BlobRegion, BlobSource};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterError {
    Blob(BlobError),
    Tensor(TensorError),
    Misaligned,
    InvalidLength,
}

impl From<BlobError> for AdapterError {
    fn from(err: BlobError) -> Self {
        AdapterError::Blob(err)
    }
}

impl From<TensorError> for AdapterError {
    fn from(err: TensorError) -> Self {
        AdapterError::Tensor(err)
    }
}

pub struct MemoryAdapter<B: BlobSource> {
    pub source: B,
}

impl<B: BlobSource> MemoryAdapter<B> {
    pub fn new(source: B) -> Self {
        Self { source }
    }

    pub fn map_f32<'a>(
        &'a self,
        region: BlobRegion,
        shape: &[usize],
        strides: &[usize],
    ) -> Result<TensorView<'a, f32>, AdapterError> {
        let bytes = self.source.map_region(region)?;
        let data = bytes_as_f32(bytes)?;
        TensorView::new(data, shape, strides).map_err(AdapterError::from)
    }
}

fn bytes_as_f32(bytes: &[u8]) -> Result<&[f32], AdapterError> {
    let align = mem::align_of::<f32>();
    let size = mem::size_of::<f32>();

    if (bytes.as_ptr() as usize) % align != 0 {
        return Err(AdapterError::Misaligned);
    }
    if bytes.len() % size != 0 {
        return Err(AdapterError::InvalidLength);
    }

    let len = bytes.len() / size;
    let ptr = bytes.as_ptr() as *const f32;
    let data = unsafe { slice::from_raw_parts(ptr, len) };
    Ok(data)
}
