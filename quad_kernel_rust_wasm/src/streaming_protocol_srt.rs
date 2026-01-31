use bytes::{Buf, BufMut, Bytes, BytesMut};
use std::sync::{Arc, Mutex};
use thiserror::Error;
use tokio::sync::mpsc;
use wasm_bindgen::prelude::*;

#[derive(Error, Debug, Clone)]
pub enum SrtError {
    #[error("Protocol violation: {0}")]
    Protocol(String),
    #[error("IO error: {0}")]
    Io(String),
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ControlPacketType {
    Handshake = 0x0,
    KeepAlive = 0x1,
    Ack = 0x2,
}

#[wasm_bindgen]
pub struct SrtSocket {
    _tx: mpsc::Sender<Vec<u8>>,
    rx: Arc<Mutex<mpsc::Receiver<Vec<u8>>>>,
}

#[wasm_bindgen]
impl SrtSocket {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let (tx, _) = mpsc::channel(100);
        let (_, rx) = mpsc::channel(100);
        Self {
            _tx: tx,
            rx: Arc::new(Mutex::new(rx)),
        }
    }

    pub async fn connect(&mut self, _addr: String) -> Result<(), JsValue> {
        // SRT Handshake simulation
        Ok(())
    }

    pub async fn send(&mut self, data: &[u8]) -> Result<(), JsValue> {
        // Send data simulation
        Ok(())
    }

    pub async fn recv(&mut self) -> Result<js_sys::Uint8Array, JsValue> {
        let mut rx = self.rx.lock().unwrap();
        if let Some(data) = rx.recv().await {
            Ok(js_sys::Uint8Array::from(&data[..]))
        } else {
            Err(JsValue::from_str("Socket closed"))
        }
    }
}

pub struct PacketHeader {
    pub timestamp: u32,
    pub dest_socket_id: u32,
}

impl PacketHeader {
    pub fn parse(buf: &mut Bytes) -> Result<Self, SrtError> {
        if buf.remaining() < 16 {
            return Err(SrtError::Protocol("Truncated header".into()));
        }
        let _first = buf.get_u32();
        let timestamp = buf.get_u32();
        let dest_socket_id = buf.get_u32();
        Ok(Self {
            timestamp,
            dest_socket_id,
        })
    }
}
