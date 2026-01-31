use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::sync::Mutex;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;

#[derive(Serialize, Deserialize, Clone)]
pub struct ProfileSample {
    pub timestamp: f64,
    pub stack: Vec<String>,
}

struct SampleBuffer {
    capacity: usize,
    buffer: VecDeque<ProfileSample>,
}

impl SampleBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: VecDeque::with_capacity(capacity),
        }
    }

    fn push(&mut self, sample: ProfileSample) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(sample);
    }

    fn take_all(&mut self) -> Vec<ProfileSample> {
        let mut out = Vec::new();
        while let Some(s) = self.buffer.pop_front() {
            out.push(s);
        }
        out
    }
}

#[wasm_bindgen]
pub struct WasmProfiler {
    buffer: Arc<Mutex<SampleBuffer>>,
    is_active: Arc<AtomicBool>,
    interval_ms: u64,
}

#[wasm_bindgen]
impl WasmProfiler {
    #[wasm_bindgen(constructor)]
    pub fn new(capacity: usize, interval_ms: u32) -> Self {
        Self {
            buffer: Arc::new(Mutex::new(SampleBuffer::new(capacity))),
            is_active: Arc::new(AtomicBool::new(false)),
            interval_ms: interval_ms as u64,
        }
    }

    #[wasm_bindgen(js_name = startProfiling)]
    pub async fn start_profiling(&self, stack_capture_fn: js_sys::Function) -> Result<(), JsValue> {
        if self.is_active.load(Ordering::Acquire) {
            return Err(JsValue::from_str("Profiler already running"));
        }

        self.is_active.store(true, Ordering::Release);
        let is_active = self.is_active.clone();
        let buffer = self.buffer.clone();
        let interval_ms = self.interval_ms;

        spawn_local(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(interval_ms));
            while is_active.load(Ordering::Acquire) {
                interval.tick().await;

                // Call JS capture function
                if let Ok(result) = stack_capture_fn.call0(&JsValue::UNDEFINED) {
                    if let Ok(stack) = serde_wasm_bindgen::from_value::<Vec<String>>(result) {
                        let sample = ProfileSample {
                            timestamp: js_sys::Date::now(),
                            stack,
                        };
                        buffer.lock().await.push(sample);
                    }
                }
            }
        });

        Ok(())
    }

    #[wasm_bindgen(js_name = stopProfiling)]
    pub async fn stop_profiling(&self) -> Result<Vec<u8>, JsValue> {
        self.is_active.store(false, Ordering::Release);
        let mut lock = self.buffer.lock().await;
        let samples = lock.take_all();
        bincode::serialize(&samples).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
