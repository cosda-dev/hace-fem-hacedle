use super::MmbDataType;

pub struct DetectFormat;

impl DetectFormat {
    pub fn new() -> Self { Self {} }

    pub fn detect(bytes: &[u8]) -> MmbDataType {
        if bytes.len() < 4 {
            return MmbDataType::Unknown;
        }

        if bytes.starts_with(b"\xFF\xD8\xFF") || bytes.starts_with(b"\x89PNG") {
            return MmbDataType::Image;
        }
        if bytes.starts_with(b"GIF8") {
            return MmbDataType::Image;
        }
        if bytes.starts_with(b"ID3") || bytes.starts_with(b"RIFF") {
            return MmbDataType::Voice;
        }

        MmbDataType::Text
    }
}
