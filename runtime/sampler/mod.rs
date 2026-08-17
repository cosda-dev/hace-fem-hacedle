mod argmax;

pub use argmax::ArgMaxSampler;

pub trait Sampler {
    fn sample(&self, logits: &[f32]) -> u32;
}

impl Sampler for ArgMaxSampler {
    fn sample(&self, logits: &[f32]) -> u32 {
        ArgMaxSampler::sample(logits)
    }
}