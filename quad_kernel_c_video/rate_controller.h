#ifndef RATE_CONTROLLER_H
#define RATE_CONTROLLER_H

#include <stdbool.h>
#include <stdint.h>


typedef enum { RC_MODE_CBR = 0, RC_MODE_VBR = 1, RC_MODE_CQP = 2 } rc_mode_t;

typedef enum {
  FRAME_TYPE_I = 0,
  FRAME_TYPE_P = 1,
  FRAME_TYPE_B = 2
} frame_type_t;

typedef struct {
  rc_mode_t mode;
  uint32_t target_bps;
  uint8_t initial_qp;
  uint8_t min_qp;
  uint8_t max_qp;
  uint32_t vbv_buffer_size; // in bits
  float vbv_init_fullness;  // 0.0 to 1.0
} rc_config_t;

typedef struct {
  float buffer_fullness;
  uint8_t current_qp;
  uint64_t total_bits;
  double avg_qp;
  uint32_t frame_count;
} rc_state_t;

typedef struct rc_controller rc_controller_t;

rc_controller_t *rc_init(const rc_config_t *config);
void rc_destroy(rc_controller_t *rc);

// Returns the QP to use for the next frame
uint8_t rc_get_frame_qp(rc_controller_t *rc, frame_type_t type);

// Updates the internal model after a frame is encoded
void rc_update_after_encode(rc_controller_t *rc, size_t frame_bits,
                            frame_type_t type);

rc_state_t rc_get_state(rc_controller_t *rc);

#endif // RATE_CONTROLLER_H
