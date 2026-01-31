#ifndef ENTROPY_CODER_H
#define ENTROPY_CODER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


typedef enum { ENTROPY_MODE_CAVLC = 0, ENTROPY_MODE_CABAC = 1 } entropy_mode_t;

typedef struct {
  entropy_mode_t mode;
  bool low_latency;
} entropy_config_t;

typedef struct {
  uint8_t *data;
  size_t size;
  size_t capacity;
  uint32_t bit_buf;
  int bit_left;
} bitstream_t;

void bitstream_init(bitstream_t *bs, uint8_t *buffer, size_t capacity);
void bitstream_put_bits(bitstream_t *bs, uint32_t val, int n);
void bitstream_flush(bitstream_t *bs);

void entropy_encode_block(bitstream_t *bs, const int16_t *coeffs, int count,
                          entropy_mode_t mode);

#endif // ENTROPY_CODER_H
