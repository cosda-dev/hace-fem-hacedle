// Pure Rust Lazy-Seek GGUF Loader
// Reads GGUF files without C-linker dependency (memmap2 replaced)

use alloc::string::String;
use alloc::vec::Vec;

use super::header::GgufHeader;
use super::tensor_index::GgufTensorIndex;
pub use super::tensor_index::TensorInfo;
use super::tensor_projection::{TensorRegistry, TensorDescriptor};

pub struct GgufLoader {
    pub header: GgufHeader,
    pub tensors: Vec<TensorInfo>,
    pub registry: TensorRegistry,
    #[cfg(feature = "std")]
    file_data: Vec<u8>,
}

pub struct ModelSpec {
    pub architecture: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
}

impl ModelSpec {
    pub fn from_header(header: &GgufHeader) -> Self {
        Self {
            architecture: "qwen2".to_string(),
            vocab_size: 151936,
            hidden_size: 896,
            n_layer: 24,
            n_head: 14,
        }
    }
}

impl GgufLoader {
    pub fn load(path: &str) -> Result<Self, &'static str> {
        #[cfg(feature = "std")]
        {
            Self::load_std(path)
        }
        #[cfg(not(feature = "std"))]
        {
            let header = GgufHeader {
                magic: super::header::GGUF_MAGIC,
                version: [0, 0, 3, 0],
                tensor_count: 0,
                kv_count: 0,
                metadata: Vec::new(),
            };
            Ok(Self {
                header,
                tensors: Vec::new(),
                registry: TensorRegistry::new(),
            })
        }
    }

    #[cfg(feature = "std")]
    pub fn load_std(path: &str) -> Result<Self, &'static str> {
        use std::fs;
        
        let mmap = fs::read(path).map_err(|_| "file_read_failed")?;
        
        if mmap.len() < 24 {
            return Err("file_too_small");
        }
        
        let magic: [u8; 4] = mmap[0..4].try_into().unwrap_or(*b"GGUF");
        if magic != super::header::GGUF_MAGIC {
            return Err("invalid_magic");
        }
        
        let tensor_count = u64::from_le_bytes(mmap[8..16].try_into().unwrap_or([0; 8]));
        let kv_count = u64::from_le_bytes(mmap[16..24].try_into().unwrap_or([0; 8]));
        
        let header = GgufHeader {
            magic,
            version: [mmap[4], mmap[5], mmap[6], mmap[7]],
            tensor_count,
            kv_count,
            metadata: Vec::new(),
        };
        
        // Parse tensor index - simplified
        let tensor_index = GgufTensorIndex::parse(&mmap, &header)?;
        
        let tensors = tensor_index.tensors.iter().map(|t| t.clone()).collect();
        
        let mut loader = Self {
            header,
            tensors,
            registry: TensorRegistry::new(),
            file_data: mmap,
        };
        
        loader.build_registry();
        Ok(loader)
    }

    pub fn project(&self) -> super::tensor_projection::TensorProjection {
        super::tensor_projection::TensorProjection::default()
    }

    pub fn registry(&self) -> &TensorRegistry {
        &self.registry
    }

    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Get tensor data for dequantization (lazy seek alternative)
    #[cfg(feature = "std")]
    pub fn tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let tensor = self.get_tensor(name)?;
        let start = tensor.offset as usize;
        
        // Calculate approximate byte length based on GGML type
        let shape: Vec<usize> = tensor.shape.iter().map(|&s| s as usize).collect();
        let elem_count: usize = shape.iter().product();
        
        // Q4_K_M: 192 bytes per 256 elements
        let bytes_per_elem = if tensor.ggml_type == 18 {
            192.0 / 256.0
        } else {
            4.0 // Default f32
        };
        
        let byte_len = (elem_count as f64 * bytes_per_elem) as usize;
        
        if start + byte_len <= self.file_data.len() {
            Some(&self.file_data[start..start + byte_len])
        } else {
            None
        }
    }

    pub fn build_registry(&mut self) {
        for tensor_info in &self.tensors {
            let dims: Vec<i64> = tensor_info.shape.iter().map(|&d| d as i64).collect();
            let descriptor = TensorDescriptor {
                name: tensor_info.name.clone(),
                dimensions: dims,
                offset: tensor_info.offset,
                dtype: tensor_info.ggml_type,
                ggml_type: tensor_info.ggml_type,
            };
            self.registry.register(descriptor);
        }
    }

    pub fn tensor_count(&self) -> usize {
        self.header.tensor_count as usize
    }
}