use crate::ring_buffer_safe::RingBufferProducer;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[repr(C)]
pub struct FrameBuffer {
    data_ptr: *mut u8,
    size: usize,
    width: i32,
    height: i32,
    channels: i32,
    sample_rate: i32,
    timestamp: u64,
    flags: u32,
}

#[wasm_bindgen]
pub struct KernelBridge {
    video_queue: RingBufferProducer,
    audio_queue: RingBufferProducer,
}

#[wasm_bindgen]
impl KernelBridge {
    #[wasm_bindgen(constructor)]
    pub fn new(video_cap: usize, audio_cap: usize) -> Self {
        Self {
            video_queue: RingBufferProducer::new(video_cap),
            audio_queue: RingBufferProducer::new(audio_cap),
        }
    }

    #[wasm_bindgen(js_name = "pushVideoFrame")]
    pub fn push_video_frame(&self, data: &[u8]) -> Result<(), JsValue> {
        self.video_queue.push(data)
    }

    #[wasm_bindgen(js_name = "pushAudioFrame")]
    pub fn push_audio_frame(&self, data: &[u8]) -> Result<(), JsValue> {
        self.audio_queue.push(data)
    }
}

// FFI export for native linking (optional but good for architectural completeness)
#[no_mangle]
pub extern "C" fn rust_kernel_process(input: *mut FrameBuffer, _output: *mut FrameBuffer) -> i32 {
    if input.is_null() {
        return -1;
    }
    // Process native FrameBuffer logic here
    0
}
