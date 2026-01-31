#include "color_converter.h"
#include "entropy_coder.h"
#include "filter_chain.h"
#include "hvcal_core.h"
#include "motion_engine.h"
#include "rate_controller.h"
#include "transform_quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void test_hvcal() {
  printf("--- Testing HVCAL ---\n");
  hvcal_initialize(true);

  hvcal_device_info_t devices[MAX_DEVICES];
  int count = hvcal_enumerate_devices(devices, MAX_DEVICES);
  printf("Found %d hardware devices\n", count);

  if (count > 0) {
    hvcal_config_t cfg = {
        .width = 1920, .height = 1080, .fps = 60, .bitrate = 5000000};
    hvcal_session_t *session = hvcal_create_session(0, HVCAL_CODEC_H265, &cfg);
    if (session) {
      printf("Session created successfully\n");

      hvcal_buffer_t frame = {.fd = 10,
                              .size = 1920 * 1080 * 3 / 2,
                              .cpu_ptr = malloc(1920 * 1080 * 3 / 2)};
      hvcal_submit_frame(session, &frame);

      uint8_t *data;
      size_t size;
      hvcal_get_bitstream(session, &data, &size);
      printf("Encoded packet size: %zu bytes\n", size);

      free(frame.cpu_ptr);
      hvcal_destroy_session(session);
    }
  }
}

void test_motion() {
  printf("\n--- Testing Motion Engine ---\n");
  MECtx *ctx = me_init_ctx(1920, 1080);

  uint8_t *ref = (uint8_t *)malloc(1920 * 1080);
  memset(ref, 128, 1920 * 1080); // Gray frame
  me_set_ref_frame(ctx, 0, ref);

  uint8_t curr_blk[256];
  memset(curr_blk, 200, 256); // A bit brighter block

  MotionVector mv;
  me_estimate_block(ctx, curr_blk, 100, 100, 16, &mv);
  printf("Motion Vector: (%d, %d), Cost: %u\n", mv.mv_x >> 2, mv.mv_y >> 2,
         mv.cost);

  uint8_t pred[256];
  me_compensate_block(ctx, &mv, 16, pred);
  printf("Compensated first pixel: %u\n", pred[0]);

  free(ref);
  me_destroy_ctx(ctx);
}

void test_rc() {
  printf("\n--- Testing Rate Controller ---\n");
  rc_config_t cfg = {.mode = RC_MODE_CBR,
                     .target_bps = 10000000,
                     .vbv_buffer_size = 2000000,
                     .vbv_init_fullness = 0.5,
                     .min_qp = 10,
                     .max_qp = 51};

  rc_controller_t *rc = rc_init(&cfg);
  uint8_t qp = rc_get_frame_qp(rc, FRAME_TYPE_I);
  printf("Initial I-frame QP: %u\n", qp);

  // Simulate encoding a frame larger than target
  rc_update_after_encode(rc, 250000,
                         FRAME_TYPE_P); // ~15mbps equivalent at 60fps
  qp = rc_get_frame_qp(rc, FRAME_TYPE_P);
  printf("New QP after large frame: %u (expected increase)\n", qp);

  rc_destroy(rc);
}

void test_color() {
  printf("\n--- Testing Color Converter ---\n");
  csc_ctx_t *ctx = csc_init(640, 480);

  uint8_t *rgb_data = (uint8_t *)malloc(640 * 480 * 3);
  memset(rgb_data, 100, 640 * 480 * 3); // Gray frame

  csc_frame_t src = {.planes = {rgb_data},
                     .strides = {640 * 3},
                     .width = 640,
                     .height = 480,
                     .format = CSC_FMT_RGB24};

  uint8_t *y = malloc(640 * 480);
  uint8_t *u = malloc(320 * 240);
  uint8_t *v = malloc(320 * 240);
  csc_frame_t dst = {.planes = {y, u, v},
                     .strides = {640, 320, 320},
                     .width = 640,
                     .height = 480,
                     .format = CSC_FMT_YUV420P};

  if (csc_convert(ctx, &src, &dst) == 0) {
    printf("Color conversion successful. Y[0]=%u\n", y[0]);
  }

  free(rgb_data);
  free(y);
  free(u);
  free(v);
  csc_destroy(ctx);
}

void test_transform() {
  printf("\n--- Testing Transform & Quant ---\n");
  int16_t input[64];
  int16_t coeffs[64];
  for (int i = 0; i < 64; i++)
    input[i] = (int16_t)(i * 2);

  quant_params_t qp = {.qp_luma = 10};
  transform_quantize_8x8(input, coeffs, &qp);

  printf("DC Coefficient: %d\n", coeffs[0]);
}

void test_entropy() {
  printf("\n--- Testing Entropy Coder ---\n");
  uint8_t buffer[1024];
  bitstream_t bs;
  bitstream_init(&bs, buffer, 1024);

  int16_t coeffs[16] = {100, -50, 20, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  entropy_encode_block(&bs, coeffs, 16, ENTROPY_MODE_CAVLC);
  bitstream_flush(&bs);

  printf("Encoded %zu bytes of bitstream\n", bs.size);
}

void mockup_filter(const uint8_t *in, uint8_t *out, uint32_t w, uint32_t h,
                   void *data) {
  // Identity filter (copy)
  memcpy(out, in, w * h);
  // Add simple constant offset
  for (uint32_t i = 0; i < w * h; i++)
    out[i] = (out[i] < 245) ? out[i] + 10 : 255;
}

void test_filter() {
  printf("\n--- Testing Filter Chain ---\n");
  vfc_config_t cfg = {.width = 160, .height = 120};
  vfc_ctx_t *ctx = vfc_init(&cfg);

  vfc_register_filter(ctx, mockup_filter, 0, NULL);

  uint8_t in[160 * 120];
  uint8_t out[160 * 120];
  memset(in, 100, 160 * 120);

  vfc_process_frame(ctx, in, out);
  printf("Filter output first pixel: %u (expected 110)\n", out[0]);

  vfc_destroy(ctx);
}

int main() {
  test_hvcal();
  test_motion();
  test_rc();
  test_color();
  test_transform();
  test_entropy();
  test_filter();
  return 0;
}
