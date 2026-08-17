use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;
use alloc::string::ToString;

use super::header::GGUF_MAGIC;

#[cfg(feature = "std")]
use std::fs::File;
#[cfg(feature = "std")]
use std::io::Read;
#[cfg(feature = "std")]
use std::path::Path;

#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub architecture: String,
    pub context_length: u64,
    pub embedding_length: u64,
    pub block_count: u64,
    pub attention_head_count: u64,
    pub tensor_names: Vec<String>,
}

impl GgufMetadata {
    pub fn new() -> Self {
        Self {
            architecture: String::new(),
            context_length: 8192,
            embedding_length: 4096,
            block_count: 0,
            attention_head_count: 32,
            tensor_names: Vec::new(),
        }
    }

    /// Parse GGUF metadata key-value pairs from a file
    #[cfg(feature = "std")]
    pub fn parse_from_file(path: &str) -> Result<Vec<(String, String)>, &'static str> {
        // Path validation to prevent directory traversal
        let path = Path::new(path);
        if !path.is_absolute() {
            // For simplicity, we only allow absolute paths in this implementation
            // In a production system, you might want to restrict to a specific directory
            return Err("parse_failed");
        }
        
        // Additional path safety checks
        let path_str = path.to_string_lossy();
        if path_str.contains("..") || path_str.contains("//") {
            return Err("parse_failed");
        }
        
        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return Err("parse_failed"),
        };
        
        let mut header = [0u8; 24];
        if let Err(_) = file.read_exact(&mut header) {
            return Err("parse_failed");
        }
        
        let magic = [header[0], header[1], header[2], header[3]];
        if magic != GGUF_MAGIC {
            return Err("parse_failed");
        }
        
        let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        if version != 3 {
            return Err("parse_failed");
        }
        
        let tensor_count = u64::from_le_bytes([
            header[8], header[9], header[10], header[11],
            header[12], header[13], header[14], header[15]
        ]);
        
        let kv_count = u64::from_le_bytes([
            header[16], header[17], header[18], header[19],
            header[20], header[21], header[22], header[23]
        ]) as usize;
        
        // Bounds checking for kv_count to prevent resource exhaustion
        if kv_count > 10000 {
            return Err("parse_failed");
        }
        
        let mut metadata = Vec::with_capacity(kv_count);
        
        for _ in 0..kv_count {
            let mut key_len = [0u8; 8];
            if let Err(_) = file.read_exact(&mut key_len) {
                return Err("parse_failed");
            }
            let key_len = u64::from_le_bytes(key_len) as usize;
            
            // Bounds checking for key_len to prevent resource exhaustion
            if key_len > 1000 {
                return Err("parse_failed");
            }
            
            let mut key = vec![0u8; key_len];
            if let Err(_) = file.read_exact(&mut key) {
                return Err("parse_failed");
            }
            let key = String::from_utf8_lossy(&key).to_string();
            
            let mut type_byte = [0u8; 1];
            if let Err(_) = file.read_exact(&mut type_byte) {
                return Err("parse_failed");
            }
            
            let value = match type_byte[0] {
                0 => String::new(),
                1 => {
                    let mut buf = [0u8; 8];
                    if let Err(_) = file.read_exact(&mut buf) {
                        return Err("parse_failed");
                    }
                    u64::from_le_bytes(buf).to_string()
                }
                2 => {
                    let mut len = [0u8; 8];
                    if let Err(_) = file.read_exact(&mut len) {
                        return Err("parse_failed");
                    }
                    let len = u64::from_le_bytes(len) as usize;
                    
                    // Bounds checking for string length to prevent resource exhaustion
                    if len > 10000 {
                        return Err("parse_failed");
                    }
                    
                    let mut buf = vec![0u8; len];
                    if let Err(_) = file.read_exact(&mut buf) {
                        return Err("parse_failed");
                    }
                    String::from_utf8_lossy(&buf).to_string()
                }
                _ => String::new(),
            };
            
            metadata.push((key, value));
        }
        
        Ok(metadata)
    }
    
    #[cfg(not(feature = "std"))]
    pub fn parse_from_file(_path: &str) -> Result<Vec<(String, String)>, &'static str> {
        Err("parse_failed")
    }
}

impl Default for GgufMetadata {
    fn default() -> Self {
        Self::new()
    }
}
