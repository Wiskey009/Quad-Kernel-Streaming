#ifndef TRANSFORM_QUANT_H
#define TRANSFORM_QUANT_H

#include <stdbool.h>
#include <stdint.h>


typedef struct {
  uint16_t qp_luma;
  uint16_t qp_chroma;
  uint8_t bit_depth;
  bool hdr_mode;
} quant_params_t;

typedef struct {
  uint8_t transform_size; // 4, 8, 16, 32
} transform_config_t;

void transform_quantize_8x8(int16_t *input, int16_t *coeffs,
                            const quant_params_t *qp);
void inverse_transform_8x8(int16_t *coeffs, int16_t *output,
                           const quant_params_t *qp);

#endif // TRANSFORM_QUANT_H
