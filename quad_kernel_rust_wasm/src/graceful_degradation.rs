use std::collections::HashSet;
use wasm_bindgen::prelude::*;
use web_sys::MediaSource;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[wasm_bindgen]
pub struct MediaCodec(String);

#[wasm_bindgen]
impl MediaCodec {
    #[wasm_bindgen(constructor)]
    pub fn new(codec: String) -> Result<MediaCodec, JsValue> {
        if codec.is_empty() {
            return Err(JsValue::from_str("Codec cannot be empty"));
        }
        Ok(Self(codec))
    }

    #[wasm_bindgen(getter)]
    pub fn value(&self) -> String {
        self.0.clone()
    }
}

#[wasm_bindgen]
pub struct GracefulDegradation {
    primary: MediaCodec,
    fallbacks: Vec<MediaCodec>,
    supported_cache: Option<HashSet<String>>,
}

#[wasm_bindgen]
impl GracefulDegradation {
    #[wasm_bindgen(constructor)]
    pub fn new(primary: MediaCodec, fallbacks: Vec<MediaCodec>) -> Self {
        Self {
            primary,
            fallbacks,
            supported_cache: None,
        }
    }

    pub async fn best_codec(&mut self) -> Result<MediaCodec, JsValue> {
        self.refresh_cache().await?;
        let supported = self.supported_cache.as_ref().unwrap();

        if supported.contains(&self.primary.0) || MediaSource::is_type_supported(&self.primary.0) {
            return Ok(self.primary.clone());
        }

        for fallback in &self.fallbacks {
            if supported.contains(&fallback.0) || MediaSource::is_type_supported(&fallback.0) {
                return Ok(fallback.clone());
            }
        }

        Err(JsValue::from_str("No supported codec found"))
    }

    async fn refresh_cache(&mut self) -> Result<(), JsValue> {
        let mut supported = HashSet::new();
        let test_codecs = vec![
            "video/mp4; codecs=\"avc1.42E01E\"",
            "video/webm; codecs=\"vp8\"",
            "video/webm; codecs=\"vp9\"",
            "video/mp4; codecs=\"hev1.1.6.L93.B0\"",
        ];
        for codec in test_codecs {
            if MediaSource::is_type_supported(codec) {
                supported.insert(codec.to_string());
            }
        }
        self.supported_cache = Some(supported);
        Ok(())
    }
}
