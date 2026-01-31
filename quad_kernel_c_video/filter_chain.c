#include "filter_chain.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define MAX_FILTERS 16

typedef struct {
  vfc_filter_func_t func;
  int priority;
  void *user_data;
  bool active;
} vfc_filter_entry_t;

struct vfc_ctx {
  vfc_config_t config;
  vfc_filter_entry_t filters[MAX_FILTERS];
  int filter_count;

  // Intermediate buffer for chain processing
  uint8_t *tmp_buffer;
};

vfc_ctx_t *vfc_init(const vfc_config_t *config) {
  vfc_ctx_t *ctx = (vfc_ctx_t *)malloc(sizeof(vfc_ctx_t));
  if (!ctx)
    return NULL;

  memset(ctx, 0, sizeof(vfc_ctx_t));
  ctx->config = *config;

  // Allocate a temp buffer for multi-stage filtering (simplified to 1.5x size
  // for YUV420)
  ctx->tmp_buffer = (uint8_t *)malloc(config->width * config->height * 2);

  return ctx;
}

void vfc_destroy(vfc_ctx_t *ctx) {
  if (!ctx)
    return;
  if (ctx->tmp_buffer)
    free(ctx->tmp_buffer);
  free(ctx);
}

int vfc_register_filter(vfc_ctx_t *ctx, vfc_filter_func_t func, int priority,
                        void *user_data) {
  if (ctx->filter_count >= MAX_FILTERS)
    return -1;

  // Sort by priority on insertion (simple insertion sort)
  int pos = ctx->filter_count;
  while (pos > 0 && ctx->filters[pos - 1].priority > priority) {
    ctx->filters[pos] = ctx->filters[pos - 1];
    pos--;
  }

  ctx->filters[pos].func = func;
  ctx->filters[pos].priority = priority;
  ctx->filters[pos].user_data = user_data;
  ctx->filters[pos].active = true;
  ctx->filter_count++;

  return 0;
}

int vfc_process_frame(vfc_ctx_t *ctx, const uint8_t *input, uint8_t *output) {
  if (ctx->filter_count == 0) {
    memcpy(output, input,
           ctx->config.width * ctx->config.height); // Simplified size
    return 0;
  }

  const uint8_t *current_in = input;
  uint8_t *current_out = (ctx->filter_count > 1) ? ctx->tmp_buffer : output;

  for (int i = 0; i < ctx->filter_count; i++) {
    if (!ctx->filters[i].active)
      continue;

    // If it's the last filter, output directly to the destination
    if (i == ctx->filter_count - 1) {
      current_out = output;
    }

    ctx->filters[i].func(current_in, current_out, ctx->config.width,
                         ctx->config.height, ctx->filters[i].user_data);

    // Swap buffers for next stage
    current_in = current_out;
    current_out = (current_in == output) ? ctx->tmp_buffer : output;
  }

  return 0;
}
