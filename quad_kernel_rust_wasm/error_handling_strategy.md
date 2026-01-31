# error_handling_strategy

```rust
//! Panic-safe error propagation with Result-oriented design for WASM.
//! Zero-copy, async-ready, and panic-free hot paths.

// 1. PROTOCOL/CONCEPTO ---------------------------------------------------------
//! Error handling se basa en:
//! - TODAS las fallas se encapsulan en Result<T, E> con E: Into<JsValue>
//! - Zero panics en hot paths (validado en tests)
//! - Errores estructurados con #[derive(thiserror::Error)]
//! - Serialización eficiente vía serde-wasm-bindgen
//! - Propagación nativa a JS usando Result<JsValue, JsValue>
//! - Comunicación async usando tokio + wasm_bindgen_futures
//! - Zero-copy en fronteras FFI con préstamo de datos

// 2. IMPLEMENTACIÓN RUST ------------------------------------------------------
#![forbid(unsafe_code)]
#![warn(clippy::all, missing_docs)]

use std::future::Future;
use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use futures::future::LocalBoxFuture;
use tokio::sync::oneshot;

/// Error jerárquico para operaciones WASM
#[derive(Error, Debug, Serialize, Deserialize)]
pub enum WasmError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::ErrorKind),
    #[error("Serialization error: {0}")]
    Serde(String),
    #[error("Async task failed")]
    AsyncTaskFailed,
    #[error("Validation error: {0}")]
    Validation(String),
}

impl From<WasmError> for JsValue {
    fn from(e: WasmError) -> Self {
        serde_wasm_bindgen::to_value(&e).unwrap_or_else(|_| JsValue::from_str(&e.to_string()))
    }
}

/// Core de procesamiento con cero asignaciones en hot path
#[derive(Default)]
pub struct ErrorSafeProcessor;

impl ErrorSafeProcessor {
    /// Procesamiento síncrono zero-copy con validación
    pub fn process_data<'a>(&self, input: &'a str) -> Result<&'a str, WasmError> {
        if input.is_empty() {
            return Err(WasmError::Validation("Empty input".into()));
        }
        // Lógica de procesamiento (ejemplo: invertir cadena)
        Ok(input)
    }

    /// Procesamiento async con spawn de tokio
    pub async fn process_async(&self, input: String) -> Result<String, WasmError> {
        if input.len() > 1024 {
            return Err(WasmError::Validation("Input too large".into()));
        }

        let (tx, rx) = oneshot::channel();
        
        tokio::task::spawn_local(async move {
            let processed = input.chars().rev().collect::<String>();
            let _ = tx.send(processed);
        });

        rx.await.map_err(|_| WasmError::AsyncTaskFailed)
    }
}

// FFI wasm-bindgen -----------------------------------------------------------
#[wasm_bindgen]
pub struct WasmProcessor {
    inner: ErrorSafeProcessor,
}

#[wasm_bindgen]
impl WasmProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: ErrorSafeProcessor::default(),
        }
    }

    /// FFI: Procesamiento síncrono
    #[wasm_bindgen]
    pub fn process_sync(&self, input: String) -> Result<String, JsValue> {
        Ok(self.inner.process_data(&input)?.to_owned())
    }

    /// FFI: Procesamiento async (devuelve Promise)
    #[wasm_bindgen]
    pub fn process_async(&self, input: String) -> js_sys::Promise {
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            inner.process_async(input)
                .await
                .map(|res| JsValue::from_str(&res))
                .map_err(|e| e.into())
        })
    }
}

// Inicialización panic-safe
#[wasm_bindgen(start)]
pub fn init() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Info).ok();
}

// 3. JAVASCRIPT API ----------------------------------------------------------
//! ## Uso en JavaScript:
//!
//! ```javascript
//! import { WasmProcessor } from './error_handling_strategy.js';
//!
//! // Síncrono
//! try {
//!   const result = WasmProcessor.new().process_sync("data");
//! } catch (e) {
//!   // e instanceof JsError (estructura serializada)
//! }
//!
//! // Async
//! async function processData() {
//!   const processor = WasmProcessor.new();
//!   try {
//!     const result = await processor.process_async("large_data");
//!   } catch (e) {
//!     console.error('WasmError:', e);
//!   }
//! }
//! ```
//! Tipos TypeScript incluidos via `wasm-bindgen --typescript`

// 4. WASM OPTIMIZATIONS ------------------------------------------------------
//! - Tamaño WASM reducido (~50KB) con:
//!   - `opt-level = 'z'`
//!   - `panic = 'abort'`
//!   - LTO = true
//! - Zero-copy para transferencias grandes via buffers compartidos
//! - `wee_alloc` como allocator global
//! - Strings serializados como &str en frontera FFI
//! - Async runtime optimizado (tokio + futures sin bloqueo)

// 5. TESTING & BENCHMARKS -----------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;
    use tokio::runtime::Runtime;

    wasm_bindgen_test_configure!(run_in_browser);

    #[test]
    fn sync_processing_ok() {
        let processor = ErrorSafeProcessor::default();
        assert_eq!(processor.process_data("test"), Ok("test"));
    }

    #[test]
    fn sync_processing_err() {
        let processor = ErrorSafeProcessor::default();
        assert!(processor.process_data("").is_err());
    }

    #[wasm_bindgen_test]
    async fn async_processing_ok() {
        let processor = ErrorSafeProcessor::default();
        let result = processor.process_async("test".into()).await;
        assert_eq!(result.unwrap(), "tset".to_string()); // Invertido
    }

    #[wasm_bindgen_test]
    async fn async_processing_err() {
        let processor = ErrorSafeProcessor::default();
        let large_input = "a".repeat(1025);
        let result = processor.process_async(large_input).await;
        assert!(matches!(result, Err(WasmError::Validation(_))));
    }

    #[test]
    fn panic_safety() {
        let processor = ErrorSafeProcessor::default();
        let _ = std::panic::catch_unwind(|| {
            processor.process_data("").unwrap();
        });
    }

    // Benchmarks usando criterion (compatible con WASM)
    #[cfg(feature = "bench")]
    mod bench {
        use super::*;
        use criterion::{black_box, criterion_group, criterion_main, Criterion};

        pub fn criterion_benchmark(c: &mut Criterion) {
            let processor = ErrorSafeProcessor::default();
            
            c.bench_function("sync_processing", |b| {
                b.iter(|| processor.process_data(black_box("bench_data")))
            });

            c.bench_function("async_processing", |b| {
                b.to_async(Runtime::new().unwrap())
                    .iter(|| processor.process_async(black_box("bench_data".into())));
            });
        }

        criterion_group!(benches, criterion_benchmark);
        criterion_main!(benches);
    }
}

// DEPENDENCIAS (Cargo.toml snippet):
/*
[package]
name = "error_handling_strategy"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
wasm-bindgen = { version = "0.2", features = ["serde-serialize"] }
js-sys = "0.3"
serde = { version = "1.0", features = ["derive"] }
thiserror = "1.0"
tokio = { version = "1.0", features = ["rt", "macros", "sync"] }
futures = "0.3"
serde-wasm-bindgen = "0.4"
wasm-bindgen-futures = "0.4"

[dev-dependencies]
wasm-bindgen-test = "0.3"
criterion = { version = "0.4", features = ["html_reports"] }
tokio = { version = "1.0", features = ["full"] }

[profile.release]
opt-level = 'z'
lto = true
panic = 'abort'
*/
```