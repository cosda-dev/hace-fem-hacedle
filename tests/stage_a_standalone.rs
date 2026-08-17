//! Standalone Stage A Test - direct GGUF magic parsing
//! Does not require full hacedle module

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_magic_detection() {
        let model_path = "D:\\host\\llama-models\\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf";
        let mut file = File::open(model_path).expect("Failed to open GGUF file");
        
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).expect("Failed to read magic");
        
        assert_eq!(magic, *b"GGUF", "Model is not GGUF format");
        println!("✓ GGUF magic verified: {:?}", String::from_utf8_lossy(&magic));
    }

    #[test] 
    fn test_gguf_version() {
        let model_path = "D:\\host\\llama-models\\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf";
        let mut file = File::open(model_path).expect("Failed to open GGUF file");
        
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).expect("Failed to read magic");
        
        let mut version = [0u8; 4];
        file.read_exact(&mut version).expect("Failed to read version");
        
        println!("✓ GGUF version: {:?}", version);
    }
}