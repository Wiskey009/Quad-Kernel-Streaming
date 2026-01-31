pub mod adaptive_bitrate_algorithm;
pub mod bridge_integration;
pub mod buffer_health_monitoring;
pub mod cross_origin_security;
pub mod dash_protocol_support;
pub mod error_handling_strategy;
pub mod graceful_degradation;
pub mod memory_pool_allocator;
pub mod ring_buffer_safe;
pub mod simd_wasm_optimizations;
pub mod streaming_analytics_client;
pub mod streaming_protocol_hls;
pub mod streaming_protocol_rtmp;
pub mod streaming_protocol_srt;
pub mod thread_pool_wasm;
pub mod wasm_binding_layer;
pub mod wasm_performance_profiling;
pub mod webrtc_data_channel;
pub mod websocket_streaming;

use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn main_js() -> Result<(), JsValue> {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    web_sys::console::log_1(&"Quad Kernel WASM initialized".into());
    Ok(())
}
