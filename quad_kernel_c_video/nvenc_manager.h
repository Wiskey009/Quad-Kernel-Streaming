#ifndef NVENC_MANAGER_H
#define NVENC_MANAGER_H

#include "hvcal_core.h"

// Note: In a real environment, we would include <nvEncodeAPI.h> and <cuda.h>
// Here we define what we need to make it functional/compilable as an interface.

typedef struct {
  uint32_t width;
  uint32_t height;
  hvcal_codec_t codec;
  uint32_t fps;
  uint32_t bitrate;
} nvenc_manager_config_t;

typedef struct nvenc_manager_ctx nvenc_manager_ctx_t;

nvenc_manager_ctx_t *nvenc_manager_create(uint32_t device_id,
                                          const nvenc_manager_config_t *config);
void nvenc_manager_destroy(nvenc_manager_ctx_t *ctx);

int nvenc_manager_submit_frame(nvenc_manager_ctx_t *ctx,
                               const hvcal_buffer_t *frame);
int nvenc_manager_get_bitstream(nvenc_manager_ctx_t *ctx, uint8_t **data,
                                size_t *size);

#endif // NVENC_MANAGER_H
