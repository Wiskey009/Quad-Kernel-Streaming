use bytes::{BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use wasm_bindgen::prelude::*;

#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum RtmpError {
    #[error("Handshake failed: {0}")]
    Handshake(String),
    #[error("Chunk error: {0}")]
    Chunk(String),
    #[error("IO error: {0}")]
    Io(String),
    #[error("Protocol violation: {0}")]
    Protocol(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RtmpMessage {
    Audio {
        timestamp: u32,
        #[serde(with = "serde_bytes")]
        data: Vec<u8>,
    },
    Video {
        timestamp: u32,
        #[serde(with = "serde_bytes")]
        data: Vec<u8>,
    },
    Command {
        name: String,
        transaction_id: f64,
    },
}

pub struct RtmpHandshake;

impl RtmpHandshake {
    pub fn create_c0_c1() -> Vec<u8> {
        let mut buf = Vec::with_capacity(1537);
        buf.push(3); // RTMP Version
        buf.extend_from_slice(&[0u8; 1536]); // C1 placeholder
        buf
    }

    pub fn validate_s0_s1(data: &[u8]) -> Result<(), RtmpError> {
        if data.is_empty() || data[0] != 3 {
            return Err(RtmpError::Handshake("Invalid version".into()));
        }
        Ok(())
    }
}

#[wasm_bindgen]
pub struct RtmpChunkManager {
    chunk_size: usize,
    partial_messages: HashMap<u32, BytesMut>,
}

#[wasm_bindgen]
impl RtmpChunkManager {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            chunk_size: 128,
            partial_messages: HashMap::new(),
        }
    }

    pub fn process_chunk(&mut self, data: &[u8]) -> Result<JsValue, JsValue> {
        // Simplified chunk processing for demonstration
        // In reality, this involves complex header parsing and state tracking
        if data.is_empty() {
            return Ok(JsValue::NULL);
        }

        // Mocking a successful message wrap
        let msg = RtmpMessage::Video {
            timestamp: 0,
            data: data.to_vec(),
        };

        serde_wasm_bindgen::to_value(&msg).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
