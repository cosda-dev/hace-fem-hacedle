use alloc::vec::Vec;
use alloc::vec;
use super::MmbDataType;

pub struct RouteMmb;

impl RouteMmb {
    pub fn route(data_type: MmbDataType) -> Vec<&'static str> {
        match data_type {
            MmbDataType::Text => vec!["candle", "llama"],
            MmbDataType::Image | MmbDataType::Video => vec!["siglip", "llama"],
            MmbDataType::Voice => vec!["whisper"],
            MmbDataType::Structured => vec!["candle"],
            MmbDataType::Sensor => vec!["candle"],
            MmbDataType::Unknown => vec!["candle"],
        }
    }
}
