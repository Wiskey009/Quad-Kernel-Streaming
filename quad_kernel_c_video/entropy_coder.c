#include "entropy_coder.h"
#include <string.h>

void bitstream_init(bitstream_t *bs, uint8_t *buffer, size_t capacity) {
  bs->data = buffer;
  bs->capacity = capacity;
  bs->size = 0;
  bs->bit_buf = 0;
  bs->bit_left = 32;
}

void bitstream_put_bits(bitstream_t *bs, uint32_t val, int n) {
  if (n == 0)
    return;

  // Simple bit buffer implementation
  if (n < bs->bit_left) {
    bs->bit_buf |= (val << (bs->bit_left - n));
    bs->bit_left -= n;
  } else {
    int rem = n - bs->bit_left;
    bs->bit_buf |= (val >> rem);

    // Flush 4 bytes
    if (bs->size + 4 <= bs->capacity) {
      bs->data[bs->size++] = (bs->bit_buf >> 24) & 0xFF;
      bs->data[bs->size++] = (bs->bit_buf >> 16) & 0xFF;
      bs->data[bs->size++] = (bs->bit_buf >> 8) & 0xFF;
      bs->data[bs->size++] = (bs->bit_buf) & 0xFF;
    }

    bs->bit_buf = (val << (32 - rem));
    bs->bit_left = 32 - rem;
  }
}

void bitstream_flush(bitstream_t *bs) {
  while (bs->bit_left < 32) {
    if (bs->size < bs->capacity) {
      bs->data[bs->size++] = (bs->bit_buf >> 24) & 0xFF;
    }
    bs->bit_buf <<= 8;
    bs->bit_left += 8;
  }
}

// Exp-Golomb coding for signed integers
static void put_ue_golomb(bitstream_t *bs, uint32_t val) {
  int len = 0;
  uint32_t temp = val + 1;
  while (temp >>= 1)
    len++;

  bitstream_put_bits(bs, 0, len);
  bitstream_put_bits(bs, val + 1, len + 1);
}

static void put_se_golomb(bitstream_t *bs, int32_t val) {
  uint32_t v = (val <= 0) ? (uint32_t)(-val * 2) : (uint32_t)(val * 2 - 1);
  put_ue_golomb(bs, v);
}

void entropy_encode_block(bitstream_t *bs, const int16_t *coeffs, int count,
                          entropy_mode_t mode) {
  // Very simplified entropy coding (using Exp-Golomb for all coefficients)
  // Real CAVLC/CABAC would use context models and specific VLC tables
  for (int i = 0; i < count; i++) {
    put_se_golomb(bs, coeffs[i]);
  }
}
