use serde::{Deserialize, Serialize};
use std::{cell::RefCell, collections::VecDeque, rc::Rc, time::Duration};
use wasm_bindgen::prelude::*;
use web_sys::Performance;

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct QoEMetrics {
    pub stall_count: u32,
    pub total_stall_duration: f64,
    pub qoe_score: f64,
    #[serde(skip)]
    pub buffer_samples: Vec<f64>,
    #[serde(skip)]
    pub last_report: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PlaybackState {
    Playing,
    Stalled { since: f64 },
}

struct MonitorInner {
    buffer_levels: VecDeque<(f64, f64)>,
    current_state: PlaybackState,
    performance: Performance,
    metrics: QoEMetrics,
    stall_threshold: f64,
}

#[wasm_bindgen]
pub struct BufferHealthMonitor {
    inner: Rc<RefCell<MonitorInner>>,
}

#[wasm_bindgen]
impl BufferHealthMonitor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<BufferHealthMonitor, JsValue> {
        let performance = web_sys::window()
            .ok_or("No window")?
            .performance()
            .ok_or("Performance API unavailable")?;

        Ok(Self {
            inner: Rc::new(RefCell::new(MonitorInner {
                buffer_levels: VecDeque::with_capacity(100),
                current_state: PlaybackState::Playing,
                performance,
                metrics: QoEMetrics::default(),
                stall_threshold: 0.1,
            })),
        })
    }

    #[wasm_bindgen(js_name = updateBufferLevel)]
    pub fn update_buffer_level(&mut self, level: f64) -> Result<(), JsValue> {
        let mut inner = self.inner.borrow_mut();
        let now = inner.performance.now() / 1000.0;

        inner.buffer_levels.push_back((now, level));
        inner.metrics.buffer_samples.push(level);

        // State machine
        match inner.current_state {
            PlaybackState::Playing => {
                if level < inner.stall_threshold {
                    inner.current_state = PlaybackState::Stalled { since: now };
                    inner.metrics.stall_count += 1;
                }
            }
            PlaybackState::Stalled { since } => {
                if level >= inner.stall_threshold {
                    inner.current_state = PlaybackState::Playing;
                    inner.metrics.total_stall_duration += now - since;
                }
            }
        }

        // Periodic QoE calculation
        if let Some(last) = inner.metrics.last_report {
            if now - last >= 1.0 {
                inner.compute_qoe();
                inner.metrics.last_report = Some(now);
            }
        } else {
            inner.metrics.last_report = Some(now);
        }

        Ok(())
    }

    #[wasm_bindgen(js_name = getMetrics)]
    pub fn get_metrics(&self) -> Result<String, JsValue> {
        let inner = self.inner.borrow();
        serde_json::to_string(&inner.metrics).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl MonitorInner {
    fn compute_qoe(&mut self) {
        if self.metrics.buffer_samples.is_empty() {
            return;
        }
        let avg_buffer: f64 = self.metrics.buffer_samples.iter().sum::<f64>()
            / self.metrics.buffer_samples.len() as f64;
        let stall_penalty = (self.metrics.total_stall_duration * 10.0).min(50.0);
        self.metrics.qoe_score = (avg_buffer * 20.0 - stall_penalty).clamp(0.0, 100.0);
        self.metrics.buffer_samples.clear();
    }
}
