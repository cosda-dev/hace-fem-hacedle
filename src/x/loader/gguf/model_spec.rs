use alloc::string::String;
use alloc::string::ToString;

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub arch: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub context_length: usize,
    pub rope_theta: f32,
}

impl ModelSpec {
    pub fn from_metadata(metadata: &[(String, String)]) -> Option<Self> {
        let get_usize = |key: &str| -> usize {
            metadata.iter()
                .find(|(k, _)| k == key)
                .and_then(|(_, v)| v.parse().ok())
                .unwrap_or(0)
        };
        
        let get_f32 = |key: &str| -> f32 {
            metadata.iter()
                .find(|(k, _)| k == key)
                .and_then(|(_, v)| v.parse().ok())
                .unwrap_or(100000.0)
        };
        
        let get_string = |key: &str| -> String {
            metadata.iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
                .unwrap_or_default()
        };

        // hidden_size từ qwen2.embedding_length
        let hidden_size = get_usize("qwen2.embedding_length");
        
        Some(Self {
            arch: get_string("general.architecture"),
            vocab_size: get_usize("qwen2.tokenizer.ggml.tokens"), // hoặc chờ count sau
            hidden_size,
            n_layer: get_usize("qwen2.block_count"),
            n_head: get_usize("qwen2.attention.head_count"),
            n_kv_head: get_usize("qwen2.attention.head_count_kv").max(1),
            context_length: get_usize("qwen2.context_length"),
            rope_theta: get_f32("qwen2.rope.freq_base"),
        })
    }
    
    pub fn set_vocab_size(&mut self, size: usize) {
        self.vocab_size = size;
    }
    
    pub fn compute_head_dim(&self) -> usize {
        self.hidden_size / self.n_head
    }
}