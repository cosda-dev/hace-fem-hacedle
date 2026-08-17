mod mmb_types;
mod detect;
mod route;
pub mod projection;
pub mod stream;
pub mod sensor;

pub use mmb_types::{MmbDataType, MmbLanguage};
pub use detect::DetectFormat;
pub use route::RouteMmb;
pub use projection::{ProjectionTensor, MmbProjection};
pub use stream::{LiveStreamHandler, RealtimeHandler};
pub use sensor::{SensorStream, SensorEmbedder};

#[derive(Debug, Clone)]
pub enum MmbError {
    InvalidFormat,
    UnsupportedType,
    ProjectionFailed,
}

pub trait MmbLoader {
    fn load(&self, path: &str) -> Result<MmbProjection, MmbError>;
    fn stream(&self, stream_ptr: usize) -> Result<ProjectionTensor, MmbError>;
}
