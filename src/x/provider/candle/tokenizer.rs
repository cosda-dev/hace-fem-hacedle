use alloc::string::String;
use alloc::vec::Vec;

#[cfg(feature = "tokenizers")]
use tokenizers::Tokenizer;

/// GGUF Tokenizer certification with optional tokenizers crate integration
pub struct GGUFTokenizer {
    pub vocab: Vec<String>,
    pub merges: Vec<(u32, u32)>,
    pub special_tokens: Vec<(String, u32)>,
    #[cfg(feature = "tokenizers")]
    pub tokenizer: Option<Tokenizer>,
}

impl GGUFTokenizer {
    pub fn new() -> Self {
        #[cfg(feature = "tokenizers")]
        {
            Self {
                vocab: Vec::new(),
                merges: Vec::new(),
                special_tokens: Vec::new(),
                tokenizer: None,
            }
        }
        #[cfg(not(feature = "tokenizers"))]
        {
            Self {
                vocab: Vec::new(),
                merges: Vec::new(),
                special_tokens: Vec::new(),
            }
        }
    }

/// Load tokenizer from GGUF metadata (tok-* keys)
    pub fn load_from_gguf(&mut self, metadata: &[(String, GGUFTokenValue)]) -> Result<(), &'static str> {
        for (key, value) in metadata {
            match value {
                GGUFTokenValue::Vocab(v) => {
                    if key.starts_with("tok") {
                        // tokopedia format: tok [[0, 'Ġ'), [1, 'ello'), ...]
                        // We need to convert this to vocab index -> string mapping
                        self.vocab = v.clone();
                    }
                }
                GGUFTokenValue::Merges(m) => {
                    if key == "merges" || key.starts_with("token-merges") {
                        self.merges = m.clone();
                    }
                }
                GGUFTokenValue::SpecialTokens(s) => {
                    self.special_tokens = s.clone();
                }
                _ => {}
            }
        }
        Ok(())
    }
    
    /// Load tokenizer from HuggingFace tokenizer.json using tokenizers crate
    #[cfg(feature = "tokenizers")]
    pub fn load_from_file(&mut self, path: &str) -> Result<(), String> {
        match Tokenizer::from_file(path) {
            Ok(t) => {
                self.tokenizer = Some(t);
                Ok(())
            }
            Err(e) => Err(format!("Failed to load tokenizer: {}", e)),
        }
    }
    
    /// Simple BPE encoding using vocabulary lookup
    /// For proper BPE, use the tokenizers crate in brain/master
    pub fn encode_bpe(&self, text: &str) -> Vec<u32> {
        // First try to use tokenizers crate if available
        #[cfg(feature = "tokenizers")]
        if let Some(ref tokenizer) = self.tokenizer {
            if let Ok(encoding) = tokenizer.encode(text) {
                return encoding.get_ids().iter().map(|&id| id as u32).collect();
            }
        }
        // Fallback to vocabulary-based encoding
        if !self.vocab.is_empty() {
            self.encode_with_vocab(text)
        } else {
            // Last resort: byte-level tokenization
            text.bytes().map(|b| b as u32).collect()
        }
    }
    
    /// Encode text using vocabulary with greedy longest-match
    fn encode_with_vocab(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        
        while i < chars.len() {
            // Find longest match for substring starting at position i
            let mut best_match_len = 0;
            let mut best_token_id = 0u32;
            
            for (token_id, vocab_token) in self.vocab.iter().enumerate() {
                if vocab_token.len() <= chars.len() - i {
                    let substr: String = chars[i..i + vocab_token.len()].iter().collect();
                    if &substr == vocab_token && vocab_token.len() > best_match_len {
                        best_match_len = vocab_token.len();
                        best_token_id = token_id as u32;
                    }
                }
            }
            
            if best_match_len > 0 {
                tokens.push(best_token_id);
                i += best_match_len;
            } else {
                // Single character not in vocab - use byte value as fallback
                tokens.push(chars[i] as u32);
                i += 1;
            }
        }
        
        tokens
    }

    fn apply_bpe_merges(&self, text: &str) -> Vec<u32> {
        // Placeholder: actual BPE merge logic
        text.bytes().map(|b| b as u32).collect()
    }
}

pub trait TokenizerEngine {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, tokens: &[u32]) -> String;
}

pub struct BpeTokenizer {
    pub tokenizer: GGUFTokenizer,
}

impl BpeTokenizer {
    pub fn new() -> Self {
        Self {
            tokenizer: GGUFTokenizer::new(),
        }
    }

    pub fn load_gguf(path: &str) -> Result<Self, &'static str> {
        let mut tokenizer = GGUFTokenizer::new();
        // Placeholder: would load from GGUF file
        Ok(Self { tokenizer })
    }
    
    #[cfg(feature = "tokenizers")]
    pub fn load_tokenizer_json(path: &str) -> Result<Self, String> {
        let mut tokenizer = GGUFTokenizer::new();
        tokenizer.load_from_file(path)?;
        Ok(Self { tokenizer })
    }
}

impl Default for BpeTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenizerEngine for BpeTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode_bpe(text)
    }

    fn decode(&self, tokens: &[u32]) -> String {
        tokens.iter().map(|&t| {
            if t < self.tokenizer.vocab.len() as u32 {
                self.tokenizer.vocab[t as usize].clone()
            } else {
                String::new()
            }
        }).collect()
    }
}

/// GGUF token value types for metadata parsing
#[derive(Debug, Clone)]
pub enum GGUFTokenValue {
    Vocab(Vec<String>),
    Merges(Vec<(u32, u32)>),
    SpecialTokens(Vec<(String, u32)>),
    String(String),
    I32(i32),
}