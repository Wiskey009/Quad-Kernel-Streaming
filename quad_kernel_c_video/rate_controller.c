#include "rate_controller.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


struct rc_controller {
  rc_config_t config;
  rc_state_t state;

  // VBV model
  double vbv_bits;
  double bits_per_frame;

  // PID / Smooth adjustment
  double qp_acc;
  double last_error;
};

rc_controller_t *rc_init(const rc_config_t *config) {
  rc_controller_t *rc = (rc_controller_t *)malloc(sizeof(rc_controller_t));
  if (!rc)
    return NULL;

  memset(rc, 0, sizeof(rc_controller_t));
  rc->config = *config;

  rc->state.current_qp = config->initial_qp ? config->initial_qp : 28;
  rc->qp_acc = (double)rc->state.current_qp;

  // Initialize VBV
  rc->vbv_bits = (double)config->vbv_buffer_size * config->vbv_init_fullness;
  // Assuming 60fps for bits per frame calculation if not specified in config
  // Actual implementation would need current FPS
  rc->bits_per_frame = (double)config->target_bps / 60.0;

  return rc;
}

void rc_destroy(rc_controller_t *rc) {
  if (rc)
    free(rc);
}

uint8_t rc_get_frame_qp(rc_controller_t *rc, frame_type_t type) {
  if (rc->config.mode == RC_MODE_CQP) {
    return rc->config.initial_qp;
  }

  // For CBR/VBR, return current calculated QP
  // Adjust based on frame type
  uint8_t qp = (uint8_t)round(rc->qp_acc);
  if (type == FRAME_TYPE_I)
    qp = (qp > 2) ? qp - 2 : qp;
  else if (type == FRAME_TYPE_B)
    qp = (qp < 50) ? qp + 2 : qp;

  // Clamp to config limits
  if (qp < rc->config.min_qp)
    qp = rc->config.min_qp;
  if (qp > rc->config.max_qp && rc->config.max_qp > 0)
    qp = rc->config.max_qp;

  return qp;
}

void rc_update_after_encode(rc_controller_t *rc, size_t frame_bits,
                            frame_type_t type) {
  rc->state.total_bits += frame_bits;
  rc->state.frame_count++;

  if (rc->config.mode == RC_MODE_CQP)
    return;

  // Update VBV
  rc->vbv_bits += rc->bits_per_frame;
  rc->vbv_bits -= (double)frame_bits;

  // Prevent overflow/underflow
  if (rc->vbv_bits > rc->config.vbv_buffer_size)
    rc->vbv_bits = rc->config.vbv_buffer_size;

  // Update state stats
  rc->state.buffer_fullness =
      (float)(rc->vbv_bits / rc->config.vbv_buffer_size);

  // PID-like adjustment for CBR
  if (rc->config.mode == RC_MODE_CBR) {
    double target_buffer = rc->config.vbv_buffer_size * 0.5;
    double error = target_buffer - rc->vbv_bits;

    // Simple proportional adjustment
    // If buffer is too full (vbv_bits high, error low), decrease bits ->
    // increase QP If buffer is too empty (vbv_bits low, error high), increase
    // bits -> decrease QP
    double sensitivity = 2.0 / (double)rc->config.vbv_buffer_size;
    rc->qp_acc += error * sensitivity;

    // Anti-windup / clamping
    if (rc->qp_acc < 1.0)
      rc->qp_acc = 1.0;
    if (rc->qp_acc > 51.0)
      rc->qp_acc = 51.0;

    rc->state.current_qp = (uint8_t)round(rc->qp_acc);
  }
}

rc_state_t rc_get_state(rc_controller_t *rc) { return rc->state; }
