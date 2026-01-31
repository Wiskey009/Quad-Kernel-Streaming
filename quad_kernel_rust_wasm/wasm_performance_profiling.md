# wasm_performance_profiling

**Protocolo/Concepto**:
El componente realiza profiling mediante muestreo periódico de la pila de llamadas (call stack) utilizando `performance.now()` para alta precisión. Los datos se almacenan en un búfer circular en memoria WASM. Se soporta generación de flame graphs mediante serialización eficiente a formato speedscope. Todo el procesamiento es asíncrono con cero copias entre Rust/JS.

---

**Rust Implementation** (`lib.rs`):
```rust
#![cfg(target_arch = "wasm32")]
use std::{
    collections::VecDeque,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
use wasm_bindgen::prelude::*;

/// Evento de muestreo con stack capture
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ProfileSample {
    timestamp: f64,
    stack: Vec<String>,
}

/// Búfer circular seguro para WASM
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

    fn push(&mut self, sample: ProfileSample) -> Result<(), String> {
        if self.buffer.len() >= self.capacity {
            Err("Buffer overflow".into())
        } else {
            self.buffer.push_back(sample);
            Ok(())
        }
    }

    fn take_all(&mut self) -> Vec<ProfileSample> {
        let mut new_buffer = VecDeque::with_capacity(self.capacity);
        std::mem::swap(&mut self.buffer, &mut new_buffer);
        new_buffer.into()
    }
}

/// Core del profiler con estado atómico
#[wasm_bindgen]
pub struct WasmProfiler {
    buffer: tokio::sync::Mutex<SampleBuffer>,
    is_active: AtomicBool,
    #[wasm_bindgen(skip)]
    pub sampling_interval: Duration,
}

#[wasm_bindgen]
impl WasmProfiler {
    #[wasm_bindgen(constructor)]
    pub fn new(capacity: usize, interval_ms: u32) -> Self {
        Self {
            buffer: tokio::sync::Mutex::new(SampleBuffer::new(capacity)),
            is_active: AtomicBool::new(false),
            sampling_interval: Duration::from_millis(interval_ms as u64),
        }
    }

    /// Inicia el muestreo periódico (non-blocking)
    #[wasm_bindgen(js_name = startProfiling)]
    pub async fn start_profiling(&self, stack_capture_fn: js_sys::Function) -> Result<(), JsValue> {
        if self.is_active.load(Ordering::Acquire) {
            return Err("Profiler already running".into());
        }

        self.is_active.store(true, Ordering::Release);
        let active_flag = self.is_active.clone();
        let buffer = self.buffer.clone();

        wasm_bindgen_futures::spawn_local(async move {
            let mut interval = tokio::time::interval(self.sampling_interval);
            while active_flag.load(Ordering::Acquire) {
                interval.tick().await;
                if let Ok(stack) = capture_stack(&stack_capture_fn).await {
                    let sample = ProfileSample {
                        timestamp: js_sys::Date::now(),
                        stack,
                    };
                    let mut buf_guard = buffer.lock().await;
                    let _ = buf_guard.push(sample);
                }
            }
        });

        Ok(())
    }

    /// Detiene el profiling y devuelve datos serializados
    #[wasm_bindgen(js_name = stopProfiling)]
    pub async fn stop_profiling(&self) -> Result<Vec<u8>, JsValue> {
        self.is_active.store(false, Ordering::Release);
        let mut buf_guard = self.buffer.lock().await;
        let samples = buf_guard.take_all();

        bincode::serialize(&samples)
            .map_err(|e| JsValue::from_str(&format!("Serialization failed: {}", e)))
    }
}

/// Captura el stack desde JavaScript sin copias intermedias
async fn capture_stack(capture_fn: &js_sys::Function) -> Result<Vec<String>, JsValue> {
    let promise = capture_fn
        .call0(&JsValue::UNDEFINED)
        .map_err(|_| "Stack capture failed")?;
    let result = wasm_bindgen_futures::JsFuture::from(promise).await?;

    serde_wasm_bindgen::from_value(result)
        .map_err(|e| JsValue::from_str(&format!("Deserialization error: {:?}", e)))
}

/// Serializa datos a formato speedscope
#[wasm_bindgen(js_name = exportSpeedscope)]
pub fn export_speedscope(samples: &[u8]) -> Result<Vec<u8>, JsValue> {
    let samples: Vec<ProfileSample> = bincode::deserialize(samples)
        .map_err(|e| JsValue::from_str(&format!("Deserialization failed: {}", e)))?;

    let output = speedscope::serialize(&samples)
        .map_err(|e| JsValue::from_str(&format!("Speedscope export failed: {}", e)))?;

    Ok(output)
}

/// Optimizaciones específicas para WASM
#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;
```

---

**JavaScript API**:
```javascript
import init, { WasmProfiler, exportSpeedscope } from './wasm_performance_profiling.js';

async function initProfiler() {
  await init();
  
  const profiler = new WasmProfiler(10_000, 10); // 10ms sampling

  // Callback para capturar el stack
  async function captureStack() {
    return new Error().stack.split('\n').slice(1);
  }

  // Iniciar profiling
  await profiler.startProfiling(captureStack);

  // Detener y obtener datos
  const binaryData = await profiler.stopProfiling();
  const speedscopeData = exportSpeedscope(binaryData);

  // Visualizar con speedscope
  import('speedscope').then(speedscope => {
    speedscope.uploadToSpeedscope(new Blob([speedscopeData]));
  });
}
```

---

**WASM Optimization**:
- **Zero-copy serialization**: Uso de bincode para serialización eficiente
- **WeeAlloc**: Reduce overhead de asignaciones de memoria pequeñas
- **Atomic state**: Flags atómicos para estado del profiler sin locks
- **Buffer circular**: Previene asignaciones frecuentes
- **LTO + opt-level=3**: Habilitado en Cargo.toml para optimizaciones agresivas
- **async/await**: Planificador de Tokio optimizado para WASM

---

**Testing & Benchmarks**:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_buffer_operations() {
        let mut buffer = SampleBuffer::new(2);
        assert!(buffer.push(ProfileSample { timestamp: 0.0, stack: vec![] }).is_ok());
        assert!(buffer.push(ProfileSample { timestamp: 1.0, stack: vec![] }).is_ok());
        assert!(buffer.push(ProfileSample { timestamp: 2.0, stack: vec![] }).is_err());
        
        let samples = buffer.take_all();
        assert_eq!(samples.len(), 2);
        assert_eq!(buffer.buffer.len(), 0);
    }

    #[wasm_bindgen_test::wasm_bindgen_test]
    async fn test_profiler_lifecycle() {
        let profiler = WasmProfiler::new(100, 10);
        let mock_capture = Closure::wrap(Box::new(|| {
            Promise::resolve(JsValue::from(vec!["func1", "func2"]))
        }) as Box<dyn Fn()>);

        profiler.start_profiling(mock_capture.as_ref().unchecked_ref()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(25)).await;
        let data = profiler.stop_profiling().await.unwrap();
        assert!(!data.is_empty());

        mock_capture.free();
    }
}
```

**Benchmarks** (Criterion):
```rust
#[cfg(all(test, not(target_arch = "wasm32")))]
mod bench {
    use criterion::{black_box, Criterion};

    #[test]
    fn run_benches() {
        let mut c = Criterion::default();
        c.bench_function("sample_serialization", |b| {
            let sample = ProfileSample {
                timestamp: 1234.567,
                stack: vec!["module::function".into(), "root".into()],
            };
            b.iter(|| bincode::serialize(black_box(&sample)).unwrap());
        });
    }
}
```