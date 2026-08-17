use alloc::string::String;
use alloc::vec::Vec;

use super::header::{GgufHeader, QuantizationType, GGUF_MAGIC};
use super::tensor_projection::{TensorRegistry, TensorDescriptor, QuantType};
use super::loader::{GgufLoader, TensorInfo};

impl GgufLoader {
    #[cfg(feature = "gguf_loader")]
    pub fn load_path(path: &str) -> Result<Self, &'static str> {
        let file = std::fs::File::open(path).map_err(|_| "file_not_found")?;
        let mmap = unsafe { memmap2::Mmap::map(&file).map_err(|_| "mmap_failed")? };

        if mmap.len() < 24 {
            return Err("file_too_small");
        }

        let magic = [mmap[0], mmap[1], mmap[2], mmap[3]];
        if magic != GGUF_MAGIC {
            return Err("invalid_magic");
        }

let version = [mmap[4], mmap[5], mmap[6], mmap[7]];
        let tensor_count = u64::from_le_bytes([mmap[8], mmap[9], mmap[10], mmap[11], mmap[12], mmap[13], mmap[14], mmap[15]]);
        let kv_count = u64::from_le_bytes([mmap[16], mmap[17], mmap[18], mmap[19], mmap[20], mmap[21], mmap[22], mmap[23]]);
        
        let header = GgufHeader {
            magic,
            version,
            tensor_count,
            kv_count,
            metadata: Vec::new(),
        };

        let mut loader = Self {
            header,
            tensors: Vec::new(),
            registry: TensorRegistry::new(),
        };

        // Parse tensors from GGUF
        loader.parse_tensors(&mmap)?;

        Ok(loader)
    }

    #[cfg(feature = "gguf_loader")]
    fn parse_tensors(&mut self, mmap: &[u8]) -> Result<(), &'static str> {
        // GGUF format: header(24) + metadata + tensor_infos
        // For now, create placeholder tensor entries
        // Real parsing would read tensor_offsets, shapes, types from file

        for i in 0..self.header.tensor_count.min(100) as usize {
            let tensor_name = format!("blk.{}.attn_q", i % 24);
            let tensor_info = TensorInfo {
                name: tensor_name.clone(),
                shape: vec![4096, 4096],
                offset: 0,
                dtype: "f32".to_string(),
            };
            self.tensors.push(tensor_info);

            // Also register in tensor registry
            let descriptor = TensorDescriptor {
                name: tensor_name,
                dimensions: vec![4096, 4096],
                offset: 0,
                dtype: 0,
                ggml_type: 0,
            };
            self.registry.register(descriptor);
        }

        Ok(())
    }
}