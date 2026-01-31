#ifndef FILTER_CHAIN_H
#define FILTER_CHAIN_H

#include "hvcal_core.h" // Reuse hvcal_buffer_t and frame structures

typedef struct {
  uint32_t width;
  uint32_t height;
  hvcal_codec_t format;
} vfc_config_t;

typedef void (*vfc_filter_func_t)(const uint8_t *in, uint8_t *out,
                                  uint32_t width, uint32_t height,
                                  void *user_data);

typedef struct vfc_ctx vfc_ctx_t;

vfc_ctx_t *vfc_init(const vfc_config_t *config);
void vfc_destroy(vfc_ctx_t *ctx);

int vfc_register_filter(vfc_ctx_t *ctx, vfc_filter_func_t func, int priority,
                        void *user_data);

// Process frame through all registered filters
int vfc_process_frame(vfc_ctx_t *ctx, const uint8_t *input, uint8_t *output);

#endif // FILTER_CHAIN_H
