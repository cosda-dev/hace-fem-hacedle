#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmbDataType {
    Text,
    Voice,
    Video,
    Image,
    Structured,
    Sensor,
    Unknown,
}

impl MmbDataType {
    pub fn from_file_extension(ext: &str) -> Self {
        match ext.to_ascii_lowercase().as_str() {
            "txt" | "md" | "json" | "yaml" | "yml" | "csv" | "xlsx" | "parquet" => MmbDataType::Text,
            "jpg" | "jpeg" | "png" | "webp" | "tif" | "bmp" => MmbDataType::Image,
            "mp3" | "wav" | "flac" | "ogg" | "aac" => MmbDataType::Voice,
            "mp4" | "mkv" | "mov" | "webm" => MmbDataType::Video,
            "lidar" | "radar" | "gps" | "imu" => MmbDataType::Sensor,
            _ => MmbDataType::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmbLanguage {
    Vi,
    En,
    Zh,
    Ar,
    Fr,
    It,
    Ja,
    Ko,
    Unknown,
}

impl MmbLanguage {
    pub fn iso_code(&self) -> &'static str {
        match self {
            MmbLanguage::Vi => "vi",
            MmbLanguage::En => "en",
            MmbLanguage::Zh => "zh",
            MmbLanguage::Ar => "ar",
            MmbLanguage::Fr => "fr",
            MmbLanguage::It => "it",
            MmbLanguage::Ja => "ja",
            MmbLanguage::Ko => "ko",
            MmbLanguage::Unknown => "unknown",
        }
    }
}
