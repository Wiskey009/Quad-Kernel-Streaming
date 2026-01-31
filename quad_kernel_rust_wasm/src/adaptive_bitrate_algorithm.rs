use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use thiserror::Error;
use wasm_bindgen::prelude::*;

#[derive(Error, Debug, Serialize, Deserialize)]
pub enum AbrError {
    #[error("Insufficient data for decision")]
    InsufficientData,
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("Calculation error: {0}")]
    CalculationError(String),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AbrConfig {
    pub min_bitrate: f64,
    pub max_bitrate: f64,
    pub target_buffer: f64,
    pub buffer_low: f64,
    pub ewma_alpha: f64,
    pub max_throughput_samples: usize,
}

impl Default for AbrConfig {
    fn default() -> Self {
        Self {
            min_bitrate: 100_000.0,
            max_bitrate: 5_000_000.0,
            target_buffer: 10.0,
            buffer_low: 5.0,
            ewma_alpha: 0.2,
            max_throughput_samples: 10,
        }
    }
}

#[wasm_bindgen]
pub struct AdaptiveBitrateController {
    config: AbrConfig,
    throughput_estimator: ThroughputEstimator,
    buffer_level: f64,
    last_quality: usize,
}

#[wasm_bindgen]
impl AdaptiveBitrateController {
    #[wasm_bindgen(constructor)]
    pub fn new(config: JsValue) -> Result<AdaptiveBitrateController, JsValue> {
        let config: AbrConfig = serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

        if config.min_bitrate <= 0.0 || config.max_bitrate <= config.min_bitrate {
            return Err(JsValue::from_str("Invalid bitrate range"));
        }

        Ok(Self {
            config: config.clone(),
            throughput_estimator: ThroughputEstimator::new(
                config.ewma_alpha,
                config.max_throughput_samples,
            ),
            buffer_level: 0.0,
            last_quality: 0,
        })
    }

    #[wasm_bindgen(js_name = addNetworkSample)]
    pub fn add_network_sample(
        &mut self,
        download_time_ms: f64,
        bytes_downloaded: u32,
    ) -> Result<(), JsValue> {
        self.throughput_estimator
            .add_sample(download_time_ms, bytes_downloaded)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = updateBufferLevel)]
    pub fn update_buffer_level(&mut self, buffer_level: f64) {
        self.buffer_level = buffer_level;
    }

    #[wasm_bindgen(js_name = calculateQuality)]
    pub fn calculate_quality(&mut self, available_bitrates: &[f64]) -> Result<usize, JsValue> {
        if available_bitrates.is_empty() {
            return Err(JsValue::from_str("Empty bitrate list"));
        }

        let throughput = self
            .throughput_estimator
            .estimate_throughput()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let safe_bitrate = self.calculate_safe_bitrate(throughput);
        let buffer_level = self.buffer_level;

        let target_index = self
            .buffer_based_selection(available_bitrates, buffer_level)
            .unwrap_or_else(|_| {
                self.throughput_based_selection(available_bitrates, safe_bitrate)
                    .unwrap_or(0)
            });

        self.last_quality = self.apply_hysteresis(target_index, available_bitrates.len());
        Ok(self.last_quality)
    }

    fn buffer_based_selection(
        &self,
        bitrates: &[f64],
        buffer_level: f64,
    ) -> Result<usize, AbrError> {
        if buffer_level < self.config.buffer_low {
            return Ok(0);
        }

        let target_bitrate = bitrates
            .iter()
            .rposition(|&br| br <= self.config.target_buffer * buffer_level)
            .unwrap_or(0);

        Ok(target_bitrate)
    }

    fn throughput_based_selection(
        &self,
        bitrates: &[f64],
        safe_bitrate: f64,
    ) -> Result<usize, AbrError> {
        bitrates
            .iter()
            .rposition(|&br| br <= safe_bitrate)
            .ok_or(AbrError::CalculationError("No suitable bitrate".into()))
    }

    fn calculate_safe_bitrate(&self, throughput: f64) -> f64 {
        let safe_bitrate = throughput * 0.9;
        safe_bitrate.clamp(self.config.min_bitrate, self.config.max_bitrate)
    }

    fn apply_hysteresis(&self, target_index: usize, max_index: usize) -> usize {
        if target_index > self.last_quality {
            (self.last_quality + 1).min(max_index - 1)
        } else if target_index < self.last_quality {
            self.last_quality.saturating_sub(1)
        } else {
            target_index
        }
    }
}

struct ThroughputEstimator {
    samples: VecDeque<(f64, u32)>,
    alpha: f64,
    ewma_throughput: Option<f64>,
    max_samples: usize,
    sum_inv_throughput: f64,
}

impl ThroughputEstimator {
    fn new(alpha: f64, max_samples: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_samples),
            alpha,
            ewma_throughput: None,
            max_samples,
            sum_inv_throughput: 0.0,
        }
    }

    fn add_sample(&mut self, download_time_ms: f64, bytes_downloaded: u32) -> Result<(), AbrError> {
        if download_time_ms <= 0.0 || bytes_downloaded == 0 {
            return Err(AbrError::InvalidConfig("Invalid sample".into()));
        }

        let current_throughput = (bytes_downloaded as f64 * 8.0) / download_time_ms;
        if current_throughput <= 0.0 {
            return Err(AbrError::CalculationError("Zero throughput".into()));
        }

        if self.samples.len() >= self.max_samples {
            if let Some((old_time, old_bytes)) = self.samples.pop_front() {
                let old_throughput = (old_bytes as f64 * 8.0) / old_time;
                self.sum_inv_throughput -= 1.0 / old_throughput;
            }
        }

        self.samples.push_back((download_time_ms, bytes_downloaded));
        self.sum_inv_throughput += 1.0 / current_throughput;

        // Harmonic mean for current window: N / sum(1/T)
        let harmonic_mean = self.samples.len() as f64 / self.sum_inv_throughput;

        // EWMA update
        self.ewma_throughput = Some(match self.ewma_throughput {
            Some(prev) => prev * (1.0 - self.alpha) + harmonic_mean * self.alpha,
            None => harmonic_mean,
        });

        Ok(())
    }

    fn estimate_throughput(&self) -> Result<f64, AbrError> {
        self.ewma_throughput
            .ok_or(AbrError::InsufficientData)
            .map(|t| t * 1000.0) // Convert to bits per second (assuming bytes and ms)
    }
}
