use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct SimdProcessor {
    // We could keep state here if needed, but for now we focus on pure processing speed
}

#[wasm_bindgen]
impl SimdProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {}
    }

    /// Process data using safe auto-vectorization
    /// Performs y = x * 2.0 + 1.0. LLVM will auto-vectorize this into WASM SIMD.
    #[wasm_bindgen(js_name = "processDataF32")]
    pub fn process_data_f32(&self, data: &[f32]) -> Vec<f32> {
        data.iter().map(|&val| val * 2.0 + 1.0).collect()
    }

    /// Standard f64 processing (Auto-vectorized version)
    #[wasm_bindgen(js_name = "processDataF64")]
    pub fn process_data_f64(&self, data: &[f64]) -> Vec<f64> {
        data.iter().map(|&val| val * 2.0 + 1.0).collect()
    }
}
