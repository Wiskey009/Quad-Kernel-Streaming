#include "../include/kernel_bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// External declarations for kernel adapters
extern KernelInterface video_kernel_interface;
extern KernelInterface audio_kernel_interface;

// Ada FFI declarations (from libquad_math.a)
extern int math_kernel_initialize(void *config);
extern int math_kernel_process(FrameBuffer *input, FrameBuffer *output);
extern int math_kernel_finalize(void);

// Rust FFI declaration (from libquad_kernel_rust_wasm.a)
extern int rust_kernel_process(void *input, void *output);

// Wrapper functions for Rust
static int rust_init_wrapper(void *config) { return 0; }
static int rust_finalize_wrapper(void) { return 0; }
static int rust_process_wrapper(FrameBuffer *input, FrameBuffer *output) {
  // Cast to void* as expected by the Rust extern "C" signature we saw,
  // or properly pass FrameBuffer pointers if Rust side matches layout.
  return rust_kernel_process((void *)input, (void *)output);
}

static KernelInterface rust_kernel_interface = {.initialize = rust_init_wrapper,
                                                .process = rust_process_wrapper,
                                                .finalize =
                                                    rust_finalize_wrapper};

// Ada math kernel interface
static KernelInterface math_kernel_interface = {
    .initialize = (int (*)(void *))math_kernel_initialize,
    .process = (int (*)(FrameBuffer *, FrameBuffer *))math_kernel_process,
    .finalize = (int (*)(void))math_kernel_finalize};

static KernelInterface *video_kernel = &video_kernel_interface;
static KernelInterface *audio_kernel = &audio_kernel_interface;
static KernelInterface *math_kernel = &math_kernel_interface;
static KernelInterface *wasm_bridge = &rust_kernel_interface;

int kernel_bridge_init(void) {
  printf("[BRIDGE] Initializing Quad Kernel IPC Bridge...\n");

  if (video_kernel->initialize(NULL) != 0)
    return -1;
  if (audio_kernel->initialize(NULL) != 0)
    return -2;
  if (math_kernel->initialize(NULL) != 0)
    return -3;
  // wasm_bridge might be initialized separately by JS

  return 0;
}

int kernel_execute_pipeline(FrameBuffer *raw_video, FrameBuffer *raw_audio,
                            FrameBuffer *output_bitstream) {
  if (!raw_video || !raw_audio || !output_bitstream)
    return -1;

  // 1. Video Processing (C Kernel)
  // printf("[BRIDGE] Step 1: Video Encoding (C Kernel)\n");
  // video_kernel.process(raw_video, temp_video_encoded);

  // 2. Audio Processing (C++ Kernel)
  // printf("[BRIDGE] Step 2: Audio DSP & Encoding (C++ Kernel)\n");
  // audio_kernel.process(raw_audio, temp_audio_encoded);

  // 3. Math Validation (Ada Kernel)
  // printf("[BRIDGE] Step 3: Precision Validation (Ada Kernel)\n");
  // math_kernel.process(NULL, NULL); // Validation mode

  // 4. WASM Packetization (Rust/WASM Bridge)
  // printf("[BRIDGE] Step 4: Protocol Framing (Rust/WASM)\n");
  // wasm_bridge.process(NULL, output_bitstream);

  // Simulation of data assembly
  output_bitstream->size = raw_video->size / 10 + raw_audio->size / 5;
  output_bitstream->timestamp = raw_video->timestamp;
  output_bitstream->flags = 4; // Encoded flag

  return 0;
}

int kernel_get_metrics(KernelType kernel, KernelMetrics *metrics) {
  if (!metrics)
    return -1;

  // Mock metrics
  metrics->cpu_usage = 15.5f;
  metrics->gpu_usage = 45.0f;
  metrics->quality_score = 0.98f;
  metrics->latency_us = 1200;

  return 0;
}
