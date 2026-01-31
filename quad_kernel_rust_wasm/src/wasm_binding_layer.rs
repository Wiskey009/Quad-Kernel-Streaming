use js_sys::Uint8Array;
use serde::{de::DeserializeOwned, Serialize};
use std::marker::PhantomData;
use thiserror::Error;
use wasm_bindgen::{prelude::*, JsCast};
use web_sys::{MessageChannel, MessageEvent, MessagePort, Worker};

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

impl From<JsValue> for JsError {
    fn from(value: JsValue) -> Self {
        JsError::JsException(format!("{:?}", value))
    }
}

impl From<JsError> for JsValue {
    fn from(error: JsError) -> Self {
        serde_wasm_bindgen::to_value(&error).unwrap_or_else(|_| JsValue::from_str("Internal error"))
    }
}

/// Trait for zero-copy WASM conversions
pub trait IntoWasm: Sized {
    type Abi: Into<JsValue>;
    fn into_abi(self) -> Result<Self::Abi, JsError>;
}

/// Trait for zero-copy Rust conversions
pub trait FromWasm: Sized {
    type Abi: From<JsValue>;
    fn from_abi(abi: Self::Abi) -> Result<Self, JsError>;
}

macro_rules! impl_wasm_primitive {
    ($($t:ty),+) => {
        $(impl IntoWasm for $t {
            type Abi = JsValue;
            fn into_abi(self) -> Result<Self::Abi, JsError> {
                Ok(JsValue::from(self))
            }
        }

        impl FromWasm for $t {
            type Abi = JsValue;
            fn from_abi(abi: Self::Abi) -> Result<Self, JsError> {
                abi.as_f64()
                    .map(|f| f as $t)
                    .ok_or_else(|| JsError::TypeMismatch {
                        expected: stringify!($t).to_string(),
                        actual: "non-number".to_string(),
                    })
            }
        })+
    };
}

impl_wasm_primitive!(u8, i8, u16, i16, u32, i32, f32, f64);

impl IntoWasm for bool {
    type Abi = JsValue;
    fn into_abi(self) -> Result<Self::Abi, JsError> {
        Ok(JsValue::from(self))
    }
}

impl FromWasm for bool {
    type Abi = JsValue;
    fn from_abi(abi: Self::Abi) -> Result<Self, JsError> {
        abi.as_bool().ok_or_else(|| JsError::TypeMismatch {
            expected: "bool".to_string(),
            actual: "non-bool".to_string(),
        })
    }
}

impl IntoWasm for String {
    type Abi = Uint8Array;
    fn into_abi(self) -> Result<Self::Abi, JsError> {
        Ok(Uint8Array::from(self.as_bytes()))
    }
}

impl FromWasm for String {
    type Abi = Uint8Array;
    fn from_abi(abi: Self::Abi) -> Result<Self, JsError> {
        Ok(String::from_utf8(abi.to_vec()).map_err(|e| JsError::Deserialization(e.to_string()))?)
    }
}

#[wasm_bindgen]
pub struct WasmBuffer {
    inner: Uint8Array,
}

impl From<JsValue> for WasmBuffer {
    fn from(value: JsValue) -> Self {
        Self {
            inner: value.unchecked_into(),
        }
    }
}

#[wasm_bindgen]
impl WasmBuffer {
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[u8]) -> Self {
        Self {
            inner: Uint8Array::from(data),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Uint8Array {
        self.inner.clone()
    }
}

impl IntoWasm for Vec<u8> {
    type Abi = WasmBuffer;
    fn into_abi(self) -> Result<Self::Abi, JsError> {
        Ok(WasmBuffer::new(&self))
    }
}

impl FromWasm for Vec<u8> {
    type Abi = WasmBuffer;
    fn from_abi(abi: Self::Abi) -> Result<Self, JsError> {
        Ok(abi.inner.to_vec())
    }
}

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
    pub fn from_worker(worker: &Worker) -> Result<Self, JsError> {
        let channel =
            MessageChannel::new().map_err(|e| JsError::JsException(format!("{:?}", e)))?;
        worker
            .post_message(&channel.port1())
            .map_err(|e| JsError::JsException(format!("{:?}", e)))?;
        Ok(Self {
            tx: channel.port1(),
            rx: channel.port2(),
            _phantom: PhantomData,
        })
    }

    pub async fn send(&self, item: T) -> Result<(), E> {
        let abi = item.into_abi().map_err(E::from)?;
        self.tx
            .post_message(&abi.into())
            .map_err(|e| E::from(JsError::JsException(format!("{:?}", e))))?;
        Ok(())
    }
}

pub fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsError> {
    serde_wasm_bindgen::to_value(value).map_err(|e| JsError::Serialization(e.to_string()))
}

pub fn from_js<T: DeserializeOwned>(value: JsValue) -> Result<T, JsError> {
    serde_wasm_bindgen::from_value(value).map_err(|e| JsError::Deserialization(e.to_string()))
}
