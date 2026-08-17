use alloc::string::String;

pub mod skb;
pub mod gguf;
pub mod ail;
pub mod mmb;
pub mod sio;
pub mod lro;
pub mod kvm;
pub mod dequant;

// LoRA overlay support
pub use lro::{LroLoader, LroAdapter, LroMetadata};
pub use lro::{LroManifest, LroSeal, RuntimeAdapterEntry};

pub use gguf::{GgufHeader, GgufMetadata, QuantizationType, GgufMmap, GgufLoader, TensorInfo, LoadedModel};
pub use gguf::tensor_projection::{TensorDescriptor, TensorProjection};
pub use gguf::tensor_projection::TensorDescriptor as TPDesc;
pub use gguf::tensor_projection::TensorProjection as TP;
pub use ail::{AilLoader, AilParser, AilValidator, ExecutionContext, IntentHeader, NarrativeBlock, TechnicalBlock, MtoDictionary};
pub use mmb::{MmbDataType, MmbLanguage, DetectFormat, RouteMmb, ProjectionTensor, MmbProjection, LiveStreamHandler, SensorStream};
pub use sio::{SioLoader, SioProjection};

pub struct LoaderConfig {
    pub model_path: String,
    pub context_size: u32,
    pub batch_size: u32,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            context_size: 8192,
            batch_size: 512,
        }
    }
}

pub struct ModelLoader {
    config: LoaderConfig,
}

impl ModelLoader {
    pub fn new(config: LoaderConfig) -> Self {
        Self { config }
    }
}
