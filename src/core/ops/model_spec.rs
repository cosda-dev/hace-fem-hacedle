// ModelSpec from GGUF - Single Source of Truth for model metadata
// Alpha-3 Phase E4: GGUF Metadata Authority

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub name: String,
    
    // Architecture
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    
    // RoPE
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScaling>,
    pub rope_type: RopeType,
    
    // Context
    pub context_length: usize,
    
    // Quantization
    pub quant_version: u32,
    pub quant_type: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeType {
    Normal,
    Linear,
    Dynamic,
    Yarn,
    Llama3,
}

#[derive(Debug, Clone)]
pub struct RopeScaling {
    pub type_: String,
    pub factor: f32,
    pub original_context_length: usize,
    pub finetuned: bool,
}

impl ModelSpec {
    pub fn from_gguf(metadata: &HashMap<String, String>) -> Result<Self, String> {
        // Extract required fields
        let hidden_size = Self::parse_usize(metadata, "general.parameter_count")?;
        
        // Extract quant type
        let quant_type = metadata.get("general.quantization_version")
            .map(|v| v.clone())
            .unwrap_or("0".to_string());
        
        Ok(Self {
            name: "unknown".to_string(),
            hidden_size,
            intermediate_size: 0,
            num_hidden_layers: 0,
            num_attention_heads: 0,
            num_key_value_heads: 0,
            head_dim: 0,
            vocab_size: 0,
            rope_theta: 10000.0,
            rope_scaling: None,
            rope_type: RopeType::Normal,
            context_length: 0,
            quant_version: 0,
            quant_type,
        })
    }
    
    fn parse_usize(map: &HashMap<String, String>, key: &str) -> Result<usize, String> {
        map.get(key)
            .ok_or_else(|| format!("Missing key: {}", key))
            .and_then(|v| v.parse().map_err(|_| format!("Invalid usize for {}", key)))
    }
    
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_spec_initialization() {
        let mut metadata = HashMap::new();
        metadata.insert("general.parameter_count".to_string(), "524288000".to_string());
        
        let spec = ModelSpec::from_gguf(&metadata);
        assert!(spec.is_ok());
    }
}