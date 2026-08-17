// Brain Runtime Integration Test
// Run with: cargo test --features std --test brain_runtime_test

use hace_fem_hacedle::x::provider::candle::{InferenceEngine, TokenizerEngine, BpeTokenizer};

#[test]
fn test_tokenizer_encode() {
    let tokenizer = BpeTokenizer::new();
    let tokens = tokenizer.encode("hello world");
    assert!(!tokens.is_empty());
}

#[test]
fn test_inference_logits() {
    let engine = InferenceEngine::default();
    // Test placeholder inference
    let _logits = engine.infer_logits("test");
}

#[test]
fn test_transformer_forward() {
    use hace_fem_hacedle::x::provider::candle::Transformer24;
    let transformer = Transformer24::default();
    let _output = transformer.forward(&[]);
}