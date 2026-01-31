use async_trait::async_trait;
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use js_sys::{ArrayBuffer, Uint8Array};
use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use thiserror::Error;
use tokio::sync::{mpsc, Mutex};
use wasm_bindgen::{prelude::*, JsCast};
use web_sys::{BinaryType, ErrorEvent, MessageEvent, WebSocket};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WebSocketMessage {
    Text(String),
    Binary(Bytes),
}

#[derive(Debug, Error)]
pub enum WebSocketError {
    #[error("Connection error: {0}")]
    Connection(String),
    #[error("Protocol violation: {0}")]
    Protocol(String),
    #[error("Channel closed")]
    ChannelClosed,
    #[error("Connection closed")]
    ConnectionClosed,
}

#[async_trait(?Send)]
trait WebSocketBackend {
    async fn send(&mut self, msg: WebSocketMessage) -> Result<(), WebSocketError>;
    async fn recv(&mut self) -> Result<WebSocketMessage, WebSocketError>;
    async fn close(&mut self) -> Result<(), WebSocketError>;
}

pub struct BrowserWebSocket {
    inner: WebSocket,
    recv_queue: mpsc::UnboundedReceiver<WebSocketMessage>,
    is_closed: Arc<AtomicBool>,
}

unsafe impl Send for BrowserWebSocket {}
unsafe impl Sync for BrowserWebSocket {}

impl BrowserWebSocket {
    pub async fn connect(url: &str) -> Result<Self, WebSocketError> {
        let ws = WebSocket::new(url).map_err(|e| {
            WebSocketError::Connection(format!("Failed to create WebSocket: {:?}", e))
        })?;
        ws.set_binary_type(BinaryType::Arraybuffer);

        let (tx, recv_queue) = mpsc::unbounded_channel();
        let is_closed = Arc::new(AtomicBool::new(false));

        let onmessage = {
            let tx = tx.clone();
            let is_closed = Arc::clone(&is_closed);
            Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
                if is_closed.load(Ordering::SeqCst) {
                    return;
                }
                if let Ok(data) = event.data().dyn_into::<ArrayBuffer>() {
                    let bytes = Uint8Array::new(&data).to_vec();
                    let _ = tx.send(WebSocketMessage::Binary(Bytes::from(bytes)));
                } else if let Ok(text) = event.data().dyn_into::<js_sys::JsString>() {
                    let _ = tx.send(WebSocketMessage::Text(text.as_string().unwrap()));
                }
            })
        };
        ws.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
        onmessage.forget();

        let onerror = {
            let is_closed = Arc::clone(&is_closed);
            Closure::<dyn FnMut(ErrorEvent)>::new(move |_| {
                is_closed.store(true, Ordering::SeqCst);
            })
        };
        ws.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        onerror.forget();

        Ok(Self {
            inner: ws,
            recv_queue,
            is_closed,
        })
    }
}

#[async_trait(?Send)]
impl WebSocketBackend for BrowserWebSocket {
    async fn send(&mut self, msg: WebSocketMessage) -> Result<(), WebSocketError> {
        if self.is_closed.load(Ordering::SeqCst) {
            return Err(WebSocketError::ConnectionClosed);
        }
        match msg {
            WebSocketMessage::Text(s) => self
                .inner
                .send_with_str(&s)
                .map_err(|e| WebSocketError::Connection(format!("{:?}", e)))?,
            WebSocketMessage::Binary(b) => {
                let array = Uint8Array::from(b.as_ref());
                self.inner
                    .send_with_array_buffer(&array.buffer())
                    .map_err(|e| WebSocketError::Connection(format!("{:?}", e)))?;
            }
        }
        Ok(())
    }

    async fn recv(&mut self) -> Result<WebSocketMessage, WebSocketError> {
        self.recv_queue
            .recv()
            .await
            .ok_or(WebSocketError::ChannelClosed)
    }

    async fn close(&mut self) -> Result<(), WebSocketError> {
        self.is_closed.store(true, Ordering::SeqCst);
        self.inner
            .close()
            .map_err(|e| WebSocketError::Connection(format!("{:?}", e)))?;
        Ok(())
    }
}

#[wasm_bindgen]
pub struct WasmWebSocketStream {
    backend: Arc<Mutex<Box<dyn WebSocketBackend>>>,
}

#[wasm_bindgen]
impl WasmWebSocketStream {
    #[wasm_bindgen(constructor)]
    pub async fn new(url: String) -> Result<WasmWebSocketStream, JsValue> {
        let backend = BrowserWebSocket::connect(&url)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self {
            backend: Arc::new(Mutex::new(Box::new(backend))),
        })
    }

    pub async fn send(&self, data: JsValue) -> Result<(), JsValue> {
        let mut backend = self.backend.lock().await;
        let msg = if data.is_string() {
            WebSocketMessage::Text(data.as_string().unwrap())
        } else if data.is_instance_of::<ArrayBuffer>() {
            let bytes = Uint8Array::new(&data).to_vec();
            WebSocketMessage::Binary(Bytes::from(bytes))
        } else {
            return Err(JsValue::from_str("Unsupported type"));
        };
        backend
            .send(msg)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub async fn recv(&self) -> Result<JsValue, JsValue> {
        let mut backend = self.backend.lock().await;
        let msg = backend
            .recv()
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        match msg {
            WebSocketMessage::Text(s) => Ok(JsValue::from_str(&s)),
            WebSocketMessage::Binary(b) => {
                let array = Uint8Array::from(b.as_ref());
                Ok(array.buffer().into())
            }
        }
    }
}
