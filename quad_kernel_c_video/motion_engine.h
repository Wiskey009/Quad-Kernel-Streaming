#ifndef MOTION_ENGINE_H
#define MOTION_ENGINE_H

#include <stdbool.h>
#include <stdint.h>


#define ME_MAX_REF_FRAMES 4
#define ME_BLOCK_SIZE 16 // Default 16x16 macroblock

typedef struct {
  uint16_t x;
  uint16_t y;
  int16_t mv_x; // Quarter-pixel precision
  int16_t mv_y; // Quarter-pixel precision
  uint32_t cost;
  uint8_t ref_idx;
} MotionVector;

typedef struct {
  void *hw_ctx;
  uint32_t width;
  uint32_t height;

  struct {
    uint8_t subpel_level;
    uint8_t search_range;
    bool enable_satd;
    bool use_hierarchical;
  } config;

  uint8_t *ref_frames[ME_MAX_REF_FRAMES];
} MECtx;

MECtx *me_init_ctx(uint32_t width, uint32_t height);
void me_destroy_ctx(MECtx *ctx);

void me_set_ref_frame(MECtx *ctx, uint8_t idx, const uint8_t *y_plane);

void me_estimate_block(MECtx *ctx, const uint8_t *curr_blk, uint32_t blk_x,
                       uint32_t blk_y, uint32_t blk_size, MotionVector *result);

void me_compensate_block(MECtx *ctx, const MotionVector *mv, uint16_t blk_size,
                         uint8_t *pred_out);

#endif // MOTION_ENGINE_H
