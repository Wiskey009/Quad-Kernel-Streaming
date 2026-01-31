use js_sys::Uint8Array;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use wasm_bindgen::{prelude::*, JsCast};
use web_sys::{MessageEvent, RtcDataChannel};

#[wasm_bindgen]
pub struct WebRtcDataChannel {
    channel: RtcDataChannel,
    _tx: mpsc::UnboundedSender<Vec<u8>>,
    rx: Arc<Mutex<mpsc::UnboundedReceiver<Vec<u8>>>>,
}

#[wasm_bindgen]
impl WebRtcDataChannel {
    #[wasm_bindgen(constructor)]
    pub fn new(channel: RtcDataChannel) -> Self {
        let (in_tx, in_rx) = mpsc::unbounded_channel();
        let (out_tx, _) = mpsc::unbounded_channel::<Vec<u8>>();

        let onmessage = Closure::wrap(Box::new(move |event: MessageEvent| {
            if let Ok(data) = event.data().dyn_into::<js_sys::ArrayBuffer>() {
                let bytes = Uint8Array::new(&data).to_vec();
                let _ = in_tx.send(bytes);
            }
        }) as Box<dyn FnMut(MessageEvent)>);

        channel.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
        onmessage.forget();

        Self {
            channel,
            _tx: out_tx,
            rx: Arc::new(Mutex::new(in_rx)),
        }
    }

    pub async fn send_bytes(&self, data: &[u8]) -> Result<(), JsValue> {
        self.channel.send_with_u8_array(data)
    }

    pub async fn recv(&self) -> Result<JsValue, JsValue> {
        let mut rx = self.rx.lock().await;
        let res: Option<Vec<u8>> = rx.recv().await;
        if let Some(data) = res {
            Ok(Uint8Array::from(&data[..]).buffer().into())
        } else {
            Err(JsValue::from_str("Channel closed"))
        }
    }

    pub fn close(&self) {
        self.channel.close();
    }
}
