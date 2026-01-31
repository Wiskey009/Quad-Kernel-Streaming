use futures::channel::{mpsc, oneshot};
use js_sys::Uint8Array;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use wasm_bindgen::prelude::*;
use web_sys::{MessageEvent, Worker};

#[derive(Debug, Serialize, Deserialize)]
pub enum PoolError {
    WorkerInitFailed(String),
    TaskTimeout(u64),
    SerializationFailed(String),
    WorkerBusy(u32),
    PoolShutdown,
}

impl From<PoolError> for JsValue {
    fn from(e: PoolError) -> Self {
        JsValue::from_str(&format!("{:?}", e))
    }
}

#[derive(Serialize, Deserialize)]
struct TaskMessage {
    task_id: u64,
    #[serde(with = "serde_bytes")]
    payload: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
struct ResultMessage {
    task_id: u64,
    #[serde(with = "serde_bytes")]
    result: Vec<u8>,
}

struct WorkerHandle {
    worker: Worker,
    _sender: mpsc::UnboundedSender<Vec<u8>>,
    busy: bool,
}

impl WorkerHandle {
    fn new(
        js_path: &str,
        pending_tasks: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Vec<u8>, PoolError>>>>>,
    ) -> Result<Self, PoolError> {
        let worker =
            Worker::new(js_path).map_err(|e| PoolError::WorkerInitFailed(format!("{:?}", e)))?;

        let (tx, _rx) = mpsc::unbounded();

        let callback = Closure::wrap(Box::new(move |e: MessageEvent| {
            if let Ok(data) = e.data().dyn_into::<Uint8Array>() {
                let bytes = data.to_vec();
                if let Ok(res) = bincode::deserialize::<ResultMessage>(&bytes) {
                    let mut lock = pending_tasks.lock().unwrap();
                    if let Some(tx) = lock.remove(&res.task_id) {
                        let _ = tx.send(Ok(res.result));
                    }
                }
            }
        }) as Box<dyn FnMut(MessageEvent)>);

        worker
            .add_event_listener_with_callback("message", callback.as_ref().unchecked_ref())
            .map_err(|e| PoolError::WorkerInitFailed(format!("{:?}", e)))?;
        callback.forget();

        Ok(Self {
            worker,
            _sender: tx,
            busy: false,
        })
    }
}

pub struct ThreadPool {
    workers: Vec<WorkerHandle>,
    task_counter: u64,
    pending_tasks: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Vec<u8>, PoolError>>>>>,
}

unsafe impl Send for ThreadPool {}
unsafe impl Sync for ThreadPool {}

impl ThreadPool {
    pub fn new(worker_count: u32, worker_js_path: &str) -> Result<ThreadPool, PoolError> {
        let pending_tasks = Arc::new(Mutex::new(HashMap::new()));
        let mut workers = Vec::with_capacity(worker_count as usize);
        for _ in 0..worker_count {
            workers.push(WorkerHandle::new(worker_js_path, pending_tasks.clone())?);
        }

        Ok(Self {
            workers,
            task_counter: 0,
            pending_tasks,
        })
    }

    pub async fn execute(&mut self, data: Vec<u8>) -> Result<Vec<u8>, PoolError> {
        let task_id = self.task_counter;
        self.task_counter += 1;

        let (tx, rx) = oneshot::channel();
        {
            let mut lock = self.pending_tasks.lock().unwrap();
            lock.insert(task_id, tx);
        }

        let worker_idx = (task_id % self.workers.len() as u64) as usize;
        let worker = &mut self.workers[worker_idx];

        let msg = TaskMessage {
            task_id,
            payload: data,
        };
        let bytes =
            bincode::serialize(&msg).map_err(|e| PoolError::SerializationFailed(e.to_string()))?;

        let array = Uint8Array::from(&bytes[..]);
        worker
            .worker
            .post_message(&array)
            .map_err(|_| PoolError::PoolShutdown)?;

        rx.await.map_err(|_| PoolError::PoolShutdown)?
    }
}

#[wasm_bindgen]
pub struct WasmThreadPool {
    inner: Arc<Mutex<ThreadPool>>,
}

#[wasm_bindgen]
impl WasmThreadPool {
    #[wasm_bindgen(constructor)]
    pub fn new(worker_count: u32, worker_js_path: String) -> Result<WasmThreadPool, JsValue> {
        let pool = ThreadPool::new(worker_count, &worker_js_path).map_err(|e| JsValue::from(e))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(pool)),
        })
    }

    pub async fn execute(&self, data: Uint8Array) -> Result<Uint8Array, JsValue> {
        let result = {
            let mut pool = self.inner.lock().unwrap();
            pool.execute(data.to_vec()).await
        };

        let res_vec = result.map_err(|e| JsValue::from(e))?;
        Ok(Uint8Array::from(&res_vec[..]))
    }
}
