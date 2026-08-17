pub struct SchedulerEngine {
    pub threads: u32,
    pub batch_size: u32,
}

impl SchedulerEngine {
    pub fn new(threads: u32, batch_size: u32) -> Self {
        Self { threads, batch_size }
    }
}

impl Default for SchedulerEngine {
    fn default() -> Self {
        Self::new(8, 512)
    }
}
