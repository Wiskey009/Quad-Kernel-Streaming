#ifndef COLOR_CONVERTER_H
#define COLOR_CONVERTER_H

#include <stddef.h>
#include <stdint.h>


typedef enum {
  CSC_FMT_RGB24,    // 8-bit RGB
  CSC_FMT_RGBA32,   // 8-bit RGBA
  CSC_FMT_RGB48,    // 16-bit RGB
  CSC_FMT_YUV420P,  // 8-bit Planar YUV
  CSC_FMT_YUV420P16 // 16-bit Planar YUV
} csc_format_t;

typedef struct {
  void *planes[4];
  size_t strides[4];
  uint32_t width;
  uint32_t height;
  csc_format_t format;
} csc_frame_t;

typedef struct csc_ctx csc_ctx_t;

csc_ctx_t *csc_init(uint32_t width, uint32_t height);
void csc_destroy(csc_ctx_t *ctx);

int csc_convert(csc_ctx_t *ctx, const csc_frame_t *src, csc_frame_t *dst);

#endif // COLOR_CONVERTER_H
