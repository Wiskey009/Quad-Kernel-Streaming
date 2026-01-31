#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Shared data structure for audio/video frames
typedef struct {
  uint8_t *data;
  size_t size;
  int width;
  int height;
  int channels;
  int sample_rate;
  uint64_t timestamp;
  uint32_t flags; // 1: Keyframe, 2: Audio, 4: Encoded
} FrameBuffer;

typedef enum {
  KERNEL_C_VIDEO = 0,
  KERNEL_CPP_AUDIO = 1,
  KERNEL_ADA_MATH = 2,
  KERNEL_RUST_WASM = 3
} KernelType;

// Standard interface that every kernel must implement for the bridge
typedef struct {
  int (*initialize)(void *config);
  int (*process)(FrameBuffer *input, FrameBuffer *output);
  int (*finalize)(void);
} KernelInterface;

// Bridge initialization
int kernel_bridge_init(void);

// Main execution pipeline: Audio + Video -> Bridge -> Final Bitstream
int kernel_execute_pipeline(FrameBuffer *raw_video, FrameBuffer *raw_audio,
                            FrameBuffer *output_bitstream);

// Telemetry and Monitoring
typedef struct {
  float cpu_usage;
  float gpu_usage;
  float quality_score;
  uint64_t latency_us;
} KernelMetrics;

int kernel_get_metrics(KernelType kernel, KernelMetrics *metrics);

#ifdef __cplusplus
}
#endif
