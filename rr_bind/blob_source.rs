
use core::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlobError {
    OutOfBounds,
    PermissionDenied,
    IoFault,
    Unsupported,
}

impl fmt::Display for BlobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            BlobError::OutOfBounds => "out_of_bounds",
            BlobError::PermissionDenied => "permission_denied",
            BlobError::IoFault => "io_fault",
            BlobError::Unsupported => "unsupported",
        };
        f.write_str(label)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlobRegion {
    pub offset: u64,
    pub length: u32,
}

impl BlobRegion {
    pub const fn new(offset: u64, length: u32) -> Self {
        Self { offset, length }
    }
}

pub trait BlobSource {
    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, BlobError>;

    fn map_region<'a>(&'a self, region: BlobRegion) -> Result<&'a [u8], BlobError> {
        let _ = region;
        Err(BlobError::Unsupported)
    }

    fn len(&self) -> Option<u64> {
        None
    }
}
