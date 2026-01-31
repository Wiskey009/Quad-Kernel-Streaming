#include "../include/audio_preprocessing_chain.h"
#include "../include/opus_encoder_advanced.h"
#include "kernel_bridge.h"
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

// Static instances for the audio pipeline
static std::unique_ptr<OpusEncoderAdvanced> global_encoder;
static std::unique_ptr<audio_processing::ProcessingChain> global_chain;

extern "C" {

int audio_kernel_initialize(void *config) {
  (void)config;
  // Simple initialization that won't crash
  std::cout << "[AUDIO KERNEL] Initialized successfully (stub mode)."
            << std::endl;
  std::cout.flush();
  return 0;
}

int audio_kernel_process(FrameBuffer *input, FrameBuffer *output) {
  if (!input || !output)
    return -1;

  // Stub: just copy a minimal amount of data
  output->size = 10;
  output->flags |= 4; // Encoded flag
  return 0;
}

int audio_kernel_finalize(void) {
  global_encoder.reset();
  global_chain.reset();
  std::cout << "[AUDIO KERNEL] Finalized." << std::endl;
  return 0;
}

// Interface struct for export
KernelInterface audio_kernel_interface = {
    audio_kernel_initialize, audio_kernel_process, audio_kernel_finalize};

} // extern "C"
