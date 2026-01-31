#include "color_converter.h"
#include <stdlib.h>
#include <string.h>

// BT.709 Fixed-point coefficients (14-bit precision)
#define R2Y 3482  // 0.2126 * 16384
#define G2Y 11718 // 0.7152 * 16384
#define B2Y 1183  // 0.0722 * 16384

struct csc_ctx {
  uint32_t width;
  uint32_t height;
};

csc_ctx_t *csc_init(uint32_t width, uint32_t height) {
  csc_ctx_t *ctx = (csc_ctx_t *)malloc(sizeof(csc_ctx_t));
  if (!ctx)
    return NULL;
  ctx->width = width;
  ctx->height = height;
  return ctx;
}

void csc_destroy(csc_ctx_t *ctx) {
  if (ctx)
    free(ctx);
}

static inline uint8_t clamp8(int val) {
  return (uint8_t)(val < 0 ? 0 : (val > 255 ? 255 : val));
}

int csc_convert(csc_ctx_t *ctx, const csc_frame_t *src, csc_frame_t *dst) {
  if (src->format == CSC_FMT_RGB24 && dst->format == CSC_FMT_YUV420P) {
    uint8_t *rgb = (uint8_t *)src->planes[0];
    uint8_t *y_plane = (uint8_t *)dst->planes[0];
    uint8_t *u_plane = (uint8_t *)dst->planes[1];
    uint8_t *v_plane = (uint8_t *)dst->planes[2];

    for (uint32_t y = 0; y < ctx->height; y++) {
      for (uint32_t x = 0; x < ctx->width; x++) {
        uint8_t r = rgb[(y * src->strides[0]) + (x * 3) + 0];
        uint8_t g = rgb[(y * src->strides[0]) + (x * 3) + 1];
        uint8_t b = rgb[(y * src->strides[0]) + (x * 3) + 2];

        // Y calculation (BT.709)
        int luma = (R2Y * r + G2Y * g + B2Y * b + 8192) >> 14;
        y_plane[y * dst->strides[0] + x] = clamp8(luma);

        // Cb, Cr calculation (for 4:2:0 subsampling, we only sample every 2x2
        // pixels)
        if ((x % 2 == 0) && (y % 2 == 0)) {
          // Simple subsampling (ideally we should average 2x2)
          float rf = r / 255.0f;
          float gf = g / 255.0f;
          float bf = b / 255.0f;
          float yf = 0.2126f * rf + 0.7152f * gf + 0.0722f * bf;
          float cb = (bf - yf) / 1.8556f + 0.5f;
          float cr = (rf - yf) / 1.5748f + 0.5f;

          u_plane[(y / 2) * dst->strides[1] + (x / 2)] =
              clamp8((int)(cb * 255.0f));
          v_plane[(y / 2) * dst->strides[2] + (x / 2)] =
              clamp8((int)(cr * 255.0f));
        }
      }
    }
    return 0;
  }

  return -1; // Format combination not supported yet
}
