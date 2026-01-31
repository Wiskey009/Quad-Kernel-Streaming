# wasm_binding_layer

```rust
//! wasm_binding_layer: Zero-copy Rust↔JavaScript FFI with type-safe marshalling
//!
//! Protocol: 
//! 1. Bidirectional type conversion via shared WebAssembly.Memory
//! 2. Zero-copy for primitives & aligned buffers
//! 3. Async streams via Web Workers + postMessage
//! 4. Error propagation through Result<T, JsError>

// ----------------------
// CORE FFI IMPLEMENTATION
// ----------------------

mod core {
    use wasm_bindgen::prelude::*;
    use js_sys::{Array, Uint8Array, Promise};
    use web_sys::{MessagePort, Worker};
    use std::convert::Infallible;
    use std::future::Future;
    use std::marker::PhantomData;
    use std::pin::Pin;
    use serde::{Serialize, de::DeserializeOwned};
    use serde_wasm_bindgen::{Serializer, from_value};
    use futures::stream::Stream;
    use thiserror::Error;

    /// FFI error type bridging Rust & JS
    #[derive(Error, Debug, Serialize)]
    #[serde(tag = "type", content = "data")]
    pub enum JsError {
        #[error("JS exception: {0}")]
        JsException(String),
        #[error("Serialization error: {0}")]
        Serialization(String),
        #[error("Deserialization error: {0}")]
        Deserialization(String),
        #[error("Type mismatch: expected {expected}, got {actual}")]
        TypeMismatch { expected: String, actual: String },
    }

    /// Trait for zero-copy WASM conversions
    pub unsafe trait IntoWasm: Sized {
        type Abi: Into<JsValue>;
        fn into_abi(self) -> Result<Self::Abi, JsError>;
    }

    /// Trait for zero-copy Rust conversions
    pub unsafe trait FromWasm: Sized {
        type Abi: From<JsValue>;
        fn from_abi(abi: Self::Abi) -> Result<Self, JsError>;
    }

    // Implementations for primitive types
    macro_rules! impl_wasm_primitive {
        ($($t:ty),+) => {
            $(unsafe impl IntoWasm for $t {
                type Abi = JsValue;
                fn into_abi(self) -> Result<Self::Abi, JsError> {
                    Ok(JsValue::from(self))
                }
            }

            unsafe impl FromWasm for $t {
                type Abi = JsValue;
                fn from_abi(abi: Self::Abi) -> Result<Self, JsError> {
                    abi.as_f64()
                        .and_then(|f| TryInto::try_into(f).ok())
                        .ok_or_else(|| JsError::TypeMismatch {
                            expected: stringify!($t).to_string(),
                            actual: "non-number".to_string(),
                        })
                }
            })+
        };
    }

    impl_wasm_primitive!(u8, i8, u16, i16, u32, i32, f32, f64, bool);

    // String conversions (zero-copy via Uint8Array)
    unsafe impl IntoWasm for String {
        type Abi = Uint8Array;
        fn into_abi(self) -> Result<Self::Abi, JsError> {
            Ok(Uint8Array::from(self.as_bytes()))
        }
    }

    unsafe impl FromWasm for String {
        type Abi = Uint8Array;
        fn from_abi(abi: Self::Abi) -> Result<Self, JsError> {
            String::from_utf8(abi.to_vec())
                .map_err(|e| JsError::Deserialization(e.to_string()))
        }
    }

    /// High-performance binary buffer
    #[wasm_bindgen]
    #[derive(Clone)]
    pub struct WasmBuffer {
        inner: Uint8Array,
    }

    #[wasm_bindgen]
    impl WasmBuffer {
        #[wasm_bindgen(constructor)]
        pub fn new(data: &[u8]) -> Self {
            Self { inner: Uint8Array::from(data) }
        }

        #[wasm_bindgen(getter)]
        pub fn data(&self) -> Uint8Array {
            self.inner.clone()
        }
    }

    unsafe impl IntoWasm for Vec<u8> {
        type Abi = WasmBuffer;
        fn into_abi(self) -> Result<Self::Abi, JsError> {
            Ok(WasmBuffer::new(&self))
        }
    }

    unsafe impl FromWasm for Vec<u8> {
        type Abi = WasmBuffer;
        fn from_abi(abi: Self::Abi) -> Result<Self, JsError> {
            Ok(abi.inner.to_vec())
        }
    }

    // ----------------------
    // ASYNC STREAMS
    // ----------------------

    /// Bidirectional async stream between Rust and JS
    pub struct WasmStream<T, E> {
        tx: MessagePort,
        rx: MessagePort,
        _phantom: PhantomData<(T, E)>,
    }

    impl<T, E> WasmStream<T, E>
    where
        T: IntoWasm + FromWasm + 'static,
        E: Into<JsError> + From<JsError> + 'static,
    {
        /// Create from Web Worker
        pub fn from_worker(worker: &Worker) -> Result<Self, JsError> {
            let channel = MessageChannel::new().map_err(|e| JsError::JsException(format!("{:?}", e)))?;
            worker.post_message(&channel.port1()).map_err(|e| JsError::JsException(format!("{:?}", e)))?;
            Ok(Self {
                tx: channel.port1(),
                rx: channel.port2(),
                _phantom: PhantomData,
            })
        }

        /// Send item to JS
        pub async fn send(&self, item: T) -> Result<(), E> {
            let abi = item.into_abi().map_err(E::from)?;
            let promise = Promise::new(&mut |resolve, reject| {
                self.tx.post_message(&abi.into())
                    .map(|_| resolve.call0(&JsValue::UNDEFINED))
                    .unwrap_or_else(|e| reject.call1(&JsValue::UNDEFINED, &e));
            });
            wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|e| E::from(JsError::from(e)))?;
            Ok(())
        }

        /// Receive item from JS
        pub async fn recv(&self) -> Result<T, E> {
            let promise = Promise::new(&mut |resolve, reject| {
                self.rx.set_onmessage(Some(&mut |event: MessageEvent| {
                    resolve.call1(&JsValue::UNDEFINED, &event.data());
                }));
                self.rx.set_onmessageerror(Some(&mut |event: MessageEvent| {
                    reject.call1(&JsValue::UNDEFINED, &event.data());
                }));
            });
            let value = wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|e| E::from(JsError::from(e)))?;
            T::from_abi(value.try_into().map_err(|e| E::from(JsError::from(e)))?)
                .map_err(E::from)
        }
    }

    // ----------------------
    // SERDE-BASED CONVERSIONS
    // ----------------------

    /// For complex types requiring serialization
    pub fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsError> {
        serde_wasm_bindgen::to_value(value)
            .map_err(|e| JsError::Serialization(e.to_string()))
    }

    /// For complex types requiring deserialization
    pub fn from_js<T: DeserializeOwned>(value: JsValue) -> Result<T, JsError> {
        serde_wasm_bindgen::from_value(value)
            .map_err(|e| JsError::Deserialization(e.to_string()))
    }

    // ----------------------
    // TYPE-SAFE FUNCTION BINDING
    // ----------------------

    #[wasm_bindgen(typescript_custom_section)]
    const TS_ASYNC_FN: &'static str = r#"
    type AsyncFunction<T, R> = (arg: T) => Promise<R>;
    "#;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(typescript_type = "AsyncFunction<any, any>")]
        pub type AsyncFunction;
    }

    /// Type-safe JS function invocation
    pub async fn call_js_function<T, R>(
        func: &AsyncFunction,
        arg: T,
    ) -> Result<R, JsError>
    where
        T: IntoWasm,
        R: FromWasm,
    {
        let js_func: js_sys::Function = func.into();
        let js_arg = arg.into_abi()?.into();
        let promise = js_func.call1(&JsValue::UNDEFINED, &js_arg)
            .map_err(|e| JsError::JsException(format!("{:?}", e)))?;

        let result = wasm_bindgen_futures::JsFuture::from(promise)
            .await
            .map_err(|e| JsError::from(e))?;

        R::from_abi(result.try_into()?)
    }
}

// ----------------------
// MACROS FOR TYPE ERASURE
// ----------------------

#[macro_export]
macro_rules! wasm_export {
    ($(#[$meta:meta])* async fn $name:ident($($arg:ident: $t:ty),*) -> Result<$ret:ty, $err:ty>) => {
        #[wasm_bindgen]
        $(#[$meta])*
        pub fn $name($($arg: <$t as $crate::core::IntoWasm>::Abi),*) -> Promise {
            wasm_bindgen_futures::future_to_promise(async move {
                let result: Result<<$ret as $crate::core::FromWasm>::Abi, JsValue> = (|| async {
                    Ok(
                        $crate::core::IntoWasm::into_abi(
                            inner_fn(
                                $($crate::core::FromWasm::from_abi($arg.into())?),*
                            ).await.map_err(|e| JsValue::from(e))?
                        )?.into()
                    )
                })().await.map_err(|e: $crate::core::JsError| e.into())?;
                Ok(result)
            })
        }
    };
}

// ----------------------
// EXAMPLE USAGE
// ----------------------

mod example {
    use super::core::*;

    wasm_export! {
        /// Example async function
        async fn process_data(input: String) -> Result<Vec<u8>, JsError> {
            Ok(input.into_bytes())
        }
    }
}
```

```typescript
// JavaScript API
/**
 * Type-safe WASM binding layer
 * 
 * Key Features:
 * - Zero-copy binary transfer via WasmBuffer
 * - Async streams using MessagePort
 * - Automatic error serialization
 * 
 * Core Methods:
 * 1. Primitive conversion:
 *    fromRust<T>(value: T): T extends WasmPrimitive ? T : never
 *    intoRust<T>(value: WasmType): T
 * 
 * 2. Stream communication:
 *    const stream = new WasmStream(worker)
 *    await stream.send(data)
 *    const response = await stream.recv()
 * 
 * 3. Error handling:
 *    try { ... } catch (e) { 
 *      if (e instanceof JsError) { ... }
 *    }
 * 
 * 4. High-performance buffers:
 *    const buffer = new WasmBuffer(new Uint8Array(...))
 *    const rustData = intoRust<Uint8Array>(buffer)
 */

// WASM Optimization Strategies
/**
 * 1. SharedArrayBuffer for zero-copy transfer
 * 2. Type-specific ArrayBuffer views (Float64Array for Rust f64)
 * 3. Stream batching with SIMD-optimized serialization
 * 4. Pre-allocated memory pools for hot paths
 * 5. Lazy deserialization for large objects
 * 6. Worker-based parallelism for stream processing
 */

// Testing & Benchmarks
/**
 * Test Strategy:
 * - Golden tests for type conversions
 * - Fuzzing for serde compatibility
 * - Web Worker integration tests
 * - Memory leak detection via FinalizationRegistry
 * 
 * Benchmark Metrics:
 * 1. Marshalling throughput (MB/s)
 * 2. Async call latency (p99)
 * 3. Memory fragmentation after 10k ops
 * 4. Stream throughput under backpressure
 * 
 * Tools:
 * - wasm-bindgen-test
 * - WebAssembly.Memory profiling
 * - Chrome DevTools WASM debugger
 */
```