// GGUF Truth Loader - Phase 1: Header + Metadata + Tensor Index
// CRD Directive P0-2: GGUF tensor_loader must work with real GGUF file

use alloc::string::String;
use alloc::vec::Vec;
use alloc::vec;

use super::header::{GgufHeader, QuantizationType, GGUF_MAGIC, GgufMetadata, GgufValue};

#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    pub name: String,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub dtype: QuantizationType,
    pub n_elements: usize,
}

pub struct GgufTensorLoader {
    header: GgufHeader,
    tensors: Vec<TensorDescriptor>,
    mmap_data: Option<Vec<u8>>,
}

impl GgufTensorLoader {
    pub fn from_file(path: &str) -> Result<Self, &'static str> {
        use std::fs::File;
        use std::io::{Read, Seek, SeekFrom};
        
        let mut file = File::open(path).map_err(|_| "file_not_found")?;
        let mut mmap_data = Vec::new();
        file.read_to_end(&mut mmap_data).map_err(|_| "read_failed")?;
        
        if mmap_data.len() < 24 {
            return Err("file_too_small");
        }
        
        // Parse header
        let magic: [u8; 4] = mmap_data[0..4].try_into().unwrap_or(*b"GGUF");
        if magic != GGUF_MAGIC {
            return Err("invalid_magic");
        }
        
        let version: [u8; 4] = mmap_data[4..8].try_into().map_err(|_| "version_parse")?;
        let tensor_count = u64::from_le_bytes(mmap_data[8..16].try_into().map_err(|_| "tensor_count_parse")?);
        let kv_count = u64::from_le_bytes(mmap_data[16..24].try_into().map_err(|_| "kv_count_parse")?);
        
        let header = GgufHeader {
            magic,
            version,
            tensor_count,
            kv_count,
            metadata: Vec::new(),
        };
        
        // TODO: Parse metadata KV
        // TODO: Parse tensor index
        
        Ok(Self {
            header,
            tensors: Vec::new(),
            mmap_data: Some(mmap_data),
        })
    }
    
    pub fn tensor_count(&self) -> usize {
        self.header.tensor_count as usize
    }
    
    pub fn get_tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

impl Default for GgufTensorLoader {
    fn default() -> Self {
        Self {
            header: GgufHeader {
                magic: GGUF_MAGIC,
                version: [0, 0, 0, 0],
                tensor_count: 0,
                kv_count: 0,
                metadata: Vec::new(),
            },
            tensors: Vec::new(),
            mmap_data: None,
        }
    }
}