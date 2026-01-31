use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, Response};

#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum HlsError {
    #[error("Invalid playlist header")]
    InvalidHeader,
    #[error("Invalid tag format: {0}")]
    InvalidTagFormat(String),
    #[error("Missing required attribute: {0}")]
    MissingAttribute(String),
    #[error("Network error: {0}")]
    Network(String),
    #[error("Status code: {0}")]
    StatusCode(u16),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Playlist {
    Master(MasterPlaylist),
    Media(MediaPlaylist),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterPlaylist {
    pub version: u8,
    pub variants: Vec<VariantStream>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantStream {
    pub bandwidth: u64,
    pub uri: String,
    pub resolution: Option<(u32, u32)>,
    pub codecs: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaPlaylist {
    pub version: u8,
    pub target_duration: f32,
    pub media_sequence: u64,
    pub segments: Vec<MediaSegment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaSegment {
    pub duration: f32,
    pub uri: String,
}

pub fn parse_hls(input: &str) -> Result<Playlist, HlsError> {
    let mut lines = input.lines().filter(|l| !l.is_empty());
    let first = lines.next().ok_or(HlsError::InvalidHeader)?;
    if first != "#EXTM3U" {
        return Err(HlsError::InvalidHeader);
    }

    let mut is_master = false;
    let content = input.to_string();
    if input.contains("#EXT-X-STREAM-INF") {
        is_master = true;
    }

    if is_master {
        Ok(Playlist::Master(parse_master(&content)))
    } else {
        Ok(Playlist::Media(parse_media(&content)))
    }
}

fn parse_master(input: &str) -> MasterPlaylist {
    let mut variants = Vec::new();
    let mut current_variant_info: Option<HashMap<String, String>> = None;

    for line in input.lines() {
        if line.starts_with("#EXT-X-STREAM-INF:") {
            let attr_str = &line[18..];
            let mut attrs = HashMap::new();
            for part in attr_str.split(',') {
                let mut kv = part.split('=');
                if let (Some(k), Some(v)) = (kv.next(), kv.next()) {
                    attrs.insert(k.trim().to_string(), v.trim().replace("\"", ""));
                }
            }
            current_variant_info = Some(attrs);
        } else if !line.starts_with('#') && !line.is_empty() {
            if let Some(attrs) = current_variant_info.take() {
                variants.push(VariantStream {
                    bandwidth: attrs
                        .get("BANDWIDTH")
                        .and_then(|b| b.parse().ok())
                        .unwrap_or(0),
                    uri: line.to_string(),
                    resolution: attrs.get("RESOLUTION").and_then(|r| {
                        let mut res = r.split('x');
                        if let (Some(w), Some(h)) = (res.next(), res.next()) {
                            Some((w.parse().ok()?, h.parse().ok()?))
                        } else {
                            None
                        }
                    }),
                    codecs: attrs.get("CODECS").cloned(),
                });
            }
        }
    }
    MasterPlaylist {
        version: 3,
        variants,
    }
}

fn parse_media(input: &str) -> MediaPlaylist {
    let mut segments = Vec::new();
    let mut current_duration = 0.0;

    for line in input.lines() {
        if line.starts_with("#EXTINF:") {
            let dur_str = line[8..].split(',').next().unwrap_or("0");
            current_duration = dur_str.parse().unwrap_or(0.0);
        } else if !line.starts_with('#') && !line.is_empty() {
            segments.push(MediaSegment {
                duration: current_duration,
                uri: line.to_string(),
            });
        }
    }
    MediaPlaylist {
        version: 3,
        target_duration: 10.0,
        media_sequence: 0,
        segments,
    }
}

#[wasm_bindgen]
pub struct JsHlsFetcher {
    base_url: String,
}

#[wasm_bindgen]
impl JsHlsFetcher {
    #[wasm_bindgen(constructor)]
    pub fn new(base_url: String) -> Self {
        Self { base_url }
    }

    pub async fn fetch_playlist(&self, path: String) -> Result<JsValue, JsValue> {
        let url = format!("{}/{}", self.base_url, path);
        let text = self
            .fetch_text(&url)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let playlist = parse_hls(&text).map_err(|e| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&playlist).map_err(|e| e.into())
    }

    async fn fetch_text(&self, url: &str) -> Result<String, HlsError> {
        let mut opts = web_sys::RequestInit::new();
        opts.set_method("GET");
        let request = web_sys::Request::new_with_str_and_init(url, &opts)
            .map_err(|e| HlsError::Network(format!("{:?}", e)))?;
        let window = web_sys::window().ok_or(HlsError::Network("No window".into()))?;
        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| HlsError::Network(format!("{:?}", e)))?;
        let resp: Response = resp_value
            .dyn_into()
            .map_err(|e| HlsError::Network(format!("{:?}", e)))?;

        if !resp.ok() {
            return Err(HlsError::StatusCode(resp.status()));
        }
        let text_value = JsFuture::from(
            resp.text()
                .map_err(|e| HlsError::Network(format!("{:?}", e)))?,
        )
        .await
        .map_err(|e| HlsError::Network(format!("{:?}", e)))?;
        Ok(text_value.as_string().unwrap_or_default())
    }
}
