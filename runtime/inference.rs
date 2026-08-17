use alloc::string::String;

pub struct InferenceEngine {
    pub model: String,
}

impl InferenceEngine {
    pub fn infer(&self, _prompt: &str) -> String {
        String::new()
    }
}