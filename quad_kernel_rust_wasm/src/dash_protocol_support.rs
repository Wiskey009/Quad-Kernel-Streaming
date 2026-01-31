use js_sys::Uint8Array;
use serde::Deserialize;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, Response};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct Mpd {
    pub period: Vec<Period>,
    pub media_presentation_duration: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct Period {
    pub adaptation_set: Vec<AdaptationSet>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdaptationSet {
    pub mime_type: String,
    pub representation: Vec<Representation>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Representation {
    pub id: String,
    pub bandwidth: u32,
    #[serde(rename = "BaseURL")]
    pub base_url: Option<String>,
}

#[wasm_bindgen]
pub struct DashPlayer {
    mpd: Arc<Mutex<Option<Mpd>>>,
    buffer: Arc<Mutex<VecDeque<Vec<u8>>>>,
    bandwidth: Arc<Mutex<u32>>,
}

#[wasm_bindgen]
impl DashPlayer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            mpd: Arc::new(Mutex::new(None)),
            buffer: Arc::new(Mutex::new(VecDeque::new())),
            bandwidth: Arc::new(Mutex::new(1_000_000)),
        }
    }

    pub async fn load_manifest(&self, url: String) -> Result<(), JsValue> {
        let text = self.fetch_text(&url).await?;
        let mpd: Mpd =
            quick_xml::de::from_str(&text).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let mut lock = self.mpd.lock().unwrap();
        *lock = Some(mpd);
        Ok(())
    }

    async fn fetch_text(&self, url: &str) -> Result<String, JsValue> {
        let mut opts = RequestInit::new();
        opts.method("GET");
        let request = Request::new_with_str_and_init(url, &opts)?;
        let window = web_sys::window().unwrap();
        let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
        let resp: Response = resp_value.dyn_into()?;
        let text = JsFuture::from(resp.text()?).await?;
        Ok(text.as_string().unwrap_or_default())
    }

    pub fn get_buffered_length(&self) -> f64 {
        self.buffer.lock().unwrap().len() as f64 * 2.0
    }
}
