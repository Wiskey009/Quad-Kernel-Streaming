use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[wasm_bindgen]
pub struct StreamingMetrics {
    pub bitrate: f64,
    pub latency_ms: f64,
    pub packet_loss: f64,
    pub jitter: f64,
    pub frame_rate: f64,
    pub buffer_level: f64,
}

#[wasm_bindgen]
impl StreamingMetrics {
    #[wasm_bindgen(constructor)]
    pub fn new(
        bitrate: f64,
        latency_ms: f64,
        packet_loss: f64,
        jitter: f64,
        frame_rate: f64,
        buffer_level: f64,
    ) -> Self {
        Self {
            bitrate,
            latency_ms,
            packet_loss,
            jitter,
            frame_rate,
            buffer_level,
        }
    }
}

#[wasm_bindgen]
pub struct StreamingAnalyticsClient {
    state: Arc<ClientState>,
    event_tx: mpsc::Sender<AnalyticsEvent>,
}

#[wasm_bindgen]
impl StreamingAnalyticsClient {
    #[wasm_bindgen(constructor)]
    pub fn new(buffer_size: usize, time_window_ms: u32) -> Self {
        let (event_tx, event_rx) = mpsc::channel(1024);
        let state = Arc::new(ClientState::new(buffer_size, time_window_ms));

        spawn_local(Self::event_processing_loop(state.clone(), event_rx));

        Self { state, event_tx }
    }

    #[wasm_bindgen(js_name = "processStreamChunk")]
    pub async fn process_stream_chunk(
        &mut self,
        data: Vec<u8>,
        timestamp: f64,
    ) -> Result<(), JsValue> {
        self.event_tx
            .send(AnalyticsEvent::DataChunk { data, timestamp })
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = "getCurrentMetrics")]
    pub async fn get_current_metrics(&self) -> Result<JsValue, JsValue> {
        let metrics = self.state.current_metrics().await;
        serde_wasm_bindgen::to_value(&metrics).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    async fn event_processing_loop(
        state: Arc<ClientState>,
        mut event_rx: mpsc::Receiver<AnalyticsEvent>,
    ) {
        while let Some(event) = event_rx.recv().await {
            match event {
                AnalyticsEvent::DataChunk { data, timestamp } => {
                    state.process_chunk(data, timestamp).await
                }
                AnalyticsEvent::Flush => state.flush_metrics().await,
            }
        }
    }
}

struct ClientState {
    metrics_buffer: RwLock<VecDeque<StreamingMetrics>>,
    buffer_size: usize,
    time_window: Duration,
    total_bytes: AtomicU32,
    chunk_count: AtomicU32,
}

impl ClientState {
    fn new(buffer_size: usize, time_window_ms: u32) -> Self {
        Self {
            metrics_buffer: RwLock::new(VecDeque::with_capacity(buffer_size)),
            buffer_size,
            time_window: Duration::from_millis(time_window_ms.into()),
            total_bytes: AtomicU32::new(0),
            chunk_count: AtomicU32::new(0),
        }
    }

    async fn process_chunk(&self, data: Vec<u8>, _timestamp: f64) {
        let byte_count = data.len() as u32;
        self.total_bytes.fetch_add(byte_count, Ordering::SeqCst);
        self.chunk_count.fetch_add(1, Ordering::SeqCst);

        let metrics = StreamingMetrics {
            bitrate: self.calculate_bitrate().await,
            ..Default::default()
        };

        let mut buffer = self.metrics_buffer.write().await;
        if buffer.len() >= self.buffer_size {
            buffer.pop_front();
        }
        buffer.push_back(metrics);
    }

    async fn calculate_bitrate(&self) -> f64 {
        let total_bytes = self.total_bytes.load(Ordering::SeqCst) as f64;
        let time_sec = self.time_window.as_secs_f64();
        if time_sec > 0.0 {
            (total_bytes * 8.0) / time_sec
        } else {
            0.0
        }
    }

    async fn current_metrics(&self) -> StreamingMetrics {
        self.metrics_buffer
            .read()
            .await
            .back()
            .cloned()
            .unwrap_or_default()
    }

    async fn flush_metrics(&self) {
        self.metrics_buffer.write().await.clear();
        self.total_bytes.store(0, Ordering::SeqCst);
        self.chunk_count.store(0, Ordering::SeqCst);
    }
}

enum AnalyticsEvent {
    DataChunk { data: Vec<u8>, timestamp: f64 },
    Flush,
}
