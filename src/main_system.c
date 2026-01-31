#include "../quad_kernel_bridge/include/kernel_bridge.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
  printf("=== QUAD KERNEL STREAMING SYSTEM v1.0 ===\n");

  // 1. Initialize Bridge (which triggers all kernels)
  if (kernel_bridge_init() != 0) {
    printf("[ERROR] Bridge initialization failed!\n");
    return 1;
  }

  printf("[SYSTEM] Pipeline ready.\n\n");

  // 2. Simulate streaming loop
  FrameBuffer raw_video;
  raw_video.data = (uint8_t *)malloc(3840 * 2160 * 3 / 2);
  raw_video.size = 3840 * 2160 * 3 / 2;
  raw_video.timestamp = 0;
  raw_video.flags = 0;

  FrameBuffer raw_audio;
  raw_audio.data = (uint8_t *)malloc(1024);
  raw_audio.size = 1024;
  raw_audio.timestamp = 0;
  raw_audio.flags = 2;

  FrameBuffer output;
  output.data = (uint8_t *)malloc(1024 * 1024);
  output.size = 0;

  for (int i = 0; i < 60; ++i) {
    raw_video.timestamp = i * 16666; // 60fps
    raw_audio.timestamp = raw_video.timestamp;

    if (kernel_execute_pipeline(&raw_video, &raw_audio, &output) == 0) {
      // Success
    }

    if (i % 10 == 0) {
      KernelMetrics metrics;
      kernel_get_metrics(KERNEL_C_VIDEO, &metrics);
      printf("[MONITOR] CPU: %.1f%% | GPU: %.1f%% | Quality: %.2f | Latency: "
             "%llu us\n",
             metrics.cpu_usage, metrics.gpu_usage, metrics.quality_score,
             (unsigned long long)metrics.latency_us);
    }
  }

  printf("\n[SYSTEM] Simulation finished.\n");

  free(raw_video.data);
  free(raw_audio.data);
  free(output.data);
  return 0;
}
