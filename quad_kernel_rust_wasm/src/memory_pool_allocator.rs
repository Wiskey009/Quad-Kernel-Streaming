use js_sys::Uint8Array;
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};
use wasm_bindgen::prelude::*;

/// Custom error types for memory allocation
#[derive(Debug)]
pub enum PoolError {
    InvalidBlockSize(usize),
    PoolExhausted,
    ForeignBlock,
}

#[derive(Debug, Clone)]
struct MemoryBlock {
    data: Vec<u8>,
    in_use: bool,
}

#[wasm_bindgen]
#[derive(Debug)]
pub struct MemoryPool {
    blocks: Arc<Mutex<VecDeque<MemoryBlock>>>,
    block_size: usize,
    capacity: usize,
}

#[wasm_bindgen]
impl MemoryPool {
    #[wasm_bindgen(constructor)]
    pub fn new(block_size: usize, capacity: usize) -> Self {
        let mut blocks = VecDeque::with_capacity(capacity);
        for _ in 0..capacity {
            blocks.push_back(MemoryBlock {
                data: vec![0; block_size],
                in_use: false,
            });
        }

        MemoryPool {
            blocks: Arc::new(Mutex::new(blocks)),
            block_size,
            capacity,
        }
    }

    #[wasm_bindgen(js_name = allocateBlock)]
    pub fn allocate_block(&self) -> Result<Uint8Array, JsValue> {
        let mut guard = self
            .blocks
            .lock()
            .map_err(|_| JsValue::from_str("Lock poisoned"))?;

        let block = guard
            .iter_mut()
            .find(|b| !b.in_use)
            .ok_or_else(|| JsValue::from_str("Pool exhausted"))?;

        block.in_use = true;

        // Zero-copy view into WASM memory.
        // We use the constructor with offset and length which is safe in js-sys.
        let offset = block.data.as_ptr() as u32;
        let len = block.data.len() as u32;

        Ok(js_sys::Uint8Array::new_with_byte_offset_and_length(
            &wasm_bindgen::memory(),
            offset,
            len,
        ))
    }

    #[wasm_bindgen(js_name = freeBlock)]
    pub fn free_block(&self, buffer: &Uint8Array) -> Result<(), JsValue> {
        let mut guard = self
            .blocks
            .lock()
            .map_err(|_| JsValue::from_str("Lock poisoned"))?;

        // We check the offset to see if it belongs to one of our blocks
        let target_offset = buffer.byte_offset() as usize;

        for b in guard.iter_mut() {
            if b.data.as_ptr() as usize == target_offset {
                if !b.in_use {
                    return Err(JsValue::from_str("Block already freed"));
                }
                b.in_use = false;
                return Ok(());
            }
        }

        Err(JsValue::from_str("Foreign block"))
    }

    #[wasm_bindgen(js_name = getStats)]
    pub fn get_stats(&self) -> JsValue {
        let guard = self.blocks.lock().unwrap();
        let used = guard.iter().filter(|b| b.in_use).count();

        let stats = serde_json::json!({
            "capacity": self.capacity,
            "blockSize": self.block_size,
            "used": used,
            "available": self.capacity - used
        });

        serde_wasm_bindgen::to_value(&stats).unwrap()
    }
}
