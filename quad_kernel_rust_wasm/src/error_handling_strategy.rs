use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use tokio::sync::oneshot;

#[derive(Error, Debug, Serialize, Deserialize)]
pub enum WasmError {
    #[error("IO error: {0}")]
    Io(String),
    #[error("Serialization error: {0}")]
    Serde(String),
    #[error("Async task failed")]
    AsyncTaskFailed,
    #[error("Validation error: {0}")]
    Validation(String),
}

impl From<WasmError> for JsValue {
    fn from(e: WasmError) -> Self {
        serde_wasm_bindgen::to_value(&e).unwrap_or_else(|_| JsValue::from_str(&e.to_string()))
    }
}

#[derive(Default, Clone)]
pub struct ErrorSafeProcessor;

impl ErrorSafeProcessor {
    pub fn process_data<'a>(&self, input: &'a str) -> Result<&'a str, WasmError> {
        if input.is_empty() {
            return Err(WasmError::Validation("Empty input".into()));
        }
        Ok(input)
    }

    pub async fn process_async(&self, input: String) -> Result<String, WasmError> {
        if input.len() > 1024 {
            return Err(WasmError::Validation("Input too large".into()));
        }

        let (tx, rx) = oneshot::channel();
        
        let processed = input.chars().rev().collect::<String>();
        let _ = tx.send(processed);

        rx.await.map_err(|_| WasmError::AsyncTaskFailed)
    }
}

#[wasm_bindgen]
pub struct WasmProcessor {
    inner: ErrorSafeProcessor,
}

#[wasm_bindgen]
impl WasmProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { inner: ErrorSafeProcessor::default() }
    }

    pub fn process_sync(&self, input: String) -> Result<String, JsValue> {
        Ok(self.inner.process_data(&input)?.to_owned())
    }

    pub async fn process_async(&self, input: String) -> Result<String, JsValue> {
        self.inner.process_async(input).await.map_err(|e| e.into())
    }
}
