use alloc::vec::Vec;

pub struct SensorStream;

impl SensorStream {
    pub fn new() -> Self { Self {} }
}

pub struct SensorEmbedder;

impl SensorEmbedder {
    pub fn embed(_sensor_data: &[u8]) -> Vec<f32> {
        Vec::new()
    }
}
