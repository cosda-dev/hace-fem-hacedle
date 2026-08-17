use alloc::vec::Vec;

pub struct FrameExtractor;

impl FrameExtractor {
    pub fn extract(_video_path: &str, _fps: f32) -> Vec<&'static [u8]> {
        Vec::new()
    }
}

pub struct LiveStreamHandler;

impl LiveStreamHandler {
    pub fn new() -> Self { Self {} }
}

pub struct RealtimeHandler;

impl RealtimeHandler {
    pub fn new() -> Self { Self {} }
}
