#include "nvenc_manager.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Forward declaration of internal NVIDIA structures (simulated for compilation)
typedef struct {
  void *encoder_handle;
  void *cuda_context;
  uint32_t width;
  uint32_t height;
  hvcal_codec_t codec;

  // Buffer pool for bitstream output
  uint8_t *bitstream_buffer;
  size_t bitstream_size;
} nvenc_internal_t;

struct nvenc_manager_ctx {
  nvenc_internal_t internal;
  hvcal_buffer_t surfaces[MAX_SURFACES];
  uint32_t current_surface_idx;
};

nvenc_manager_ctx_t *
nvenc_manager_create(uint32_t device_id, const nvenc_manager_config_t *config) {
  nvenc_manager_ctx_t *ctx =
      (nvenc_manager_ctx_t *)malloc(sizeof(nvenc_manager_ctx_t));
  if (!ctx)
    return NULL;

  memset(ctx, 0, sizeof(nvenc_manager_ctx_t));

  ctx->internal.width = config->width;
  ctx->internal.height = config->height;
  ctx->internal.codec = config->codec;

  printf("[NVENC] Initializing encoder on device %u for %ux%u\n", device_id,
         config->width, config->height);

  // Allocate bitstream buffer for results
  ctx->internal.bitstream_size =
      config->width * config->height; // Roughly max expected for 4k
  ctx->internal.bitstream_buffer =
      (uint8_t *)malloc(ctx->internal.bitstream_size);

  // If bitstream buffer allocation failed, cleanup and return NULL
  if (!ctx->internal.bitstream_buffer) {
    free(ctx);
    return NULL;
  }

  return ctx;
}

void nvenc_manager_destroy(nvenc_manager_ctx_t *ctx) {
  if (!ctx)
    return;

  printf("[NVENC] Destroying encoder session\n");

  if (ctx->internal.bitstream_buffer)
    free(ctx->internal.bitstream_buffer);
  free(ctx);
}

int nvenc_manager_submit_frame(nvenc_manager_ctx_t *ctx,
                               const hvcal_buffer_t *frame) {
  if (!ctx || !frame || !ctx->internal.bitstream_buffer)
    return -1;

  printf("[NVENC] Submitting frame, DMA-FD: %d\n", frame->fd);

  // Simulate encoding time/logic
  // memcpy to internal output (mock)
  if (frame->cpu_ptr) {
    size_t to_copy = (frame->size < ctx->internal.bitstream_size)
                         ? frame->size
                         : ctx->internal.bitstream_size;
    memcpy(ctx->internal.bitstream_buffer, frame->cpu_ptr, to_copy);
  }

  return 0;
}

int nvenc_manager_get_bitstream(nvenc_manager_ctx_t *ctx, uint8_t **data,
                                size_t *size) {
  if (!ctx || !data || !size || !ctx->internal.bitstream_buffer)
    return -1;

  *data = ctx->internal.bitstream_buffer;
  *size = ctx->internal.bitstream_size / 10; // Mock compressed size

  return 0;
}
