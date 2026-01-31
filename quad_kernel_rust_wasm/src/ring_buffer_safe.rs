use std::sync::{
    atomic::{AtomicU8, AtomicUsize, Ordering},
    Arc,
};
use wasm_bindgen::prelude::*;

struct RingBuffer {
    buffer: Box<[AtomicU8]>,
    capacity: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}

// AtomicU8 and AtomicUsize are Send + Sync, so RingBuffer is automatically Send + Sync.

impl RingBuffer {
    fn new(capacity: usize) -> Arc<Self> {
        let cap = capacity.next_power_of_two();
        let mut buffer = Vec::with_capacity(cap);
        for _ in 0..cap {
            buffer.push(AtomicU8::new(0));
        }

        Arc::new(Self {
            buffer: buffer.into_boxed_slice(),
            capacity: cap,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        })
    }

    fn try_push(&self, value: u8) -> Result<(), u8> {
        let current_head = self.head.load(Ordering::Acquire);
        let current_tail = self.tail.load(Ordering::Acquire);

        if (current_head + 1) % self.capacity == current_tail {
            return Err(value);
        }

        self.buffer[current_head].store(value, Ordering::Release);
        self.head
            .store((current_head + 1) % self.capacity, Ordering::Release);
        Ok(())
    }

    fn try_pop(&self) -> Option<u8> {
        let current_head = self.head.load(Ordering::Acquire);
        let current_tail = self.tail.load(Ordering::Acquire);

        if current_head == current_tail {
            return None;
        }

        let val = self.buffer[current_tail].load(Ordering::Acquire);
        self.tail
            .store((current_tail + 1) % self.capacity, Ordering::Release);
        Some(val)
    }

    fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        (head + self.capacity - tail) % self.capacity
    }
}

#[wasm_bindgen]
pub struct RingBufferProducer {
    buffer: Arc<RingBuffer>,
}

#[wasm_bindgen]
pub struct RingBufferConsumer {
    buffer: Arc<RingBuffer>,
}

#[wasm_bindgen]
impl RingBufferProducer {
    #[wasm_bindgen(constructor)]
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: RingBuffer::new(capacity),
        }
    }

    pub fn push(&self, data: &[u8]) -> Result<(), JsValue> {
        for byte in data {
            self.buffer
                .try_push(*byte)
                .map_err(|_| JsValue::from_str("Buffer full"))?;
        }
        Ok(())
    }

    pub fn get_consumer(&self) -> RingBufferConsumer {
        RingBufferConsumer {
            buffer: self.buffer.clone(),
        }
    }
}

#[wasm_bindgen]
impl RingBufferConsumer {
    pub fn pull(&self, len: usize) -> Vec<u8> {
        let mut result = Vec::with_capacity(len);
        for _ in 0..len {
            if let Some(byte) = self.buffer.try_pop() {
                result.push(byte);
            } else {
                break;
            }
        }
        result
    }

    pub fn available(&self) -> usize {
        self.buffer.len()
    }
}
