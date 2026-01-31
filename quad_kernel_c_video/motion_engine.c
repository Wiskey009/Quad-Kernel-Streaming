#include "motion_engine.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

MECtx *me_init_ctx(uint32_t width, uint32_t height) {
  MECtx *ctx = (MECtx *)malloc(sizeof(MECtx));
  if (!ctx)
    return NULL;

  memset(ctx, 0, sizeof(MECtx));
  ctx->width = width;
  ctx->height = height;
  ctx->config.search_range = 16;
  ctx->config.subpel_level = 2; // Half-pixel default

  return ctx;
}

void me_destroy_ctx(MECtx *ctx) {
  if (!ctx)
    return;
  for (int i = 0; i < ME_MAX_REF_FRAMES; i++) {
    if (ctx->ref_frames[i])
      free(ctx->ref_frames[i]);
  }
  free(ctx);
}

void me_set_ref_frame(MECtx *ctx, uint8_t idx, const uint8_t *y_plane) {
  if (idx >= ME_MAX_REF_FRAMES || !ctx)
    return;
  size_t size = ctx->width * ctx->height;
  if (!ctx->ref_frames[idx]) {
    ctx->ref_frames[idx] = (uint8_t *)malloc(size);
  }

  if (ctx->ref_frames[idx]) {
    memcpy(ctx->ref_frames[idx], y_plane, size);
  }
}

static uint32_t calculate_sad(const uint8_t *curr, const uint8_t *ref,
                              uint32_t stride_c, uint32_t stride_r,
                              uint32_t size) {
  uint32_t sad = 0;
  for (uint32_t y = 0; y < size; y++) {
    for (uint32_t x = 0; x < size; x++) {
      sad += abs(curr[y * stride_c + x] - ref[y * stride_r + x]);
    }
  }
  return sad;
}

void me_estimate_block(MECtx *ctx, const uint8_t *curr_blk, uint32_t blk_x,
                       uint32_t blk_y, uint32_t blk_size,
                       MotionVector *result) {
  uint32_t min_sad = 0xFFFFFFFF;
  int16_t best_mv_x = 0;
  int16_t best_mv_y = 0;

  uint8_t *ref = ctx->ref_frames[0]; // Simplified to first ref frame
  if (!ref)
    return;

  // Basic Diamond Search Pattern (Simplified for bridge implementation)
  // In actual production, this would use hierarchical ME or LDSP/SDSP
  int range = ctx->config.search_range;

  for (int dy = -range; dy <= range; dy++) {
    for (int dx = -range; dx <= range; dx++) {
      int rx = (int)blk_x + dx;
      int ry = (int)blk_y + dy;

      if (rx < 0 || ry < 0 || rx + blk_size > ctx->width ||
          ry + blk_size > ctx->height)
        continue;

      uint32_t sad = calculate_sad(curr_blk, &ref[ry * ctx->width + rx],
                                   blk_size, ctx->width, blk_size);
      if (sad < min_sad) {
        min_sad = sad;
        best_mv_x = (int16_t)dx;
        best_mv_y = (int16_t)dy;
      }
    }
  }

  result->x = (uint16_t)blk_x;
  result->y = (uint16_t)blk_y;
  result->mv_x = best_mv_x << 2; // Convert to quarter-pel
  result->mv_y = best_mv_y << 2;
  result->cost = min_sad;
  result->ref_idx = 0;
}

// 6-tap Wiener filter implementation for half-pixel interpolation
// [-1, 4, -10, 58, 17, -5] >> 6
static uint8_t interpolate_6tap(const uint8_t *src, int stride) {
  static const int filter[6] = {-1, 4, -10, 58, 17, -5};
  int val = 0;
  for (int i = 0; i < 6; i++) {
    val += src[(i - 2) * stride] * filter[i];
  }
  val = (val + 32) >> 6;
  return (uint8_t)(val < 0 ? 0 : (val > 255 ? 255 : val));
}

void me_compensate_block(MECtx *ctx, const MotionVector *mv, uint16_t blk_size,
                         uint8_t *pred_out) {
  uint8_t *ref = ctx->ref_frames[mv->ref_idx];
  if (!ref)
    return;

  int int_x = (int)mv->x + (mv->mv_x >> 2);
  int int_y = (int)mv->y + (mv->mv_y >> 2);

  // Very simplified compensation (integer pel only for now in this snippet)
  // Full sub-pel would require multi-pass horizontal/vertical filter
  for (uint16_t y = 0; y < blk_size; y++) {
    for (uint16_t x = 0; x < blk_size; x++) {
      int rx = int_x + x;
      int ry = int_y + y;
      if (rx >= 0 && ry >= 0 && rx < ctx->width && ry < ctx->height) {
        pred_out[y * blk_size + x] = ref[ry * ctx->width + rx];
      } else {
        pred_out[y * blk_size + x] = 128; // Padding
      }
    }
  }
}
