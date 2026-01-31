#include "../quad_kernel_bridge/include/kernel_bridge.h"
#include "hvcal_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Static state for the video kernel
static hvcal_session_t *global_session = NULL;
static hvcal_config_t global_config;

int video_kernel_initialize(void *config) {
  if (hvcal_initialize(false) != 0) {
    fprintf(stderr, "[VIDEO KERNEL] Failed to initialize HVCAL core\n");
    return -1;
  }

  // Default configuration for 4K streaming
  global_config.width = 3840;
  global_config.height = 2160;
  global_config.fps = 60;
  global_config.bitrate = 20000000; // 20 Mbps
  global_config.gop_size = 60;
  global_config.bit_depth = 8;
  global_config.hdr = false;

  // Use device 0 by default, H265 codec
  global_session = hvcal_create_session(0, HVCAL_CODEC_H265, &global_config);

  if (!global_session) {
    fprintf(stderr, "[VIDEO KERNEL] Failed to create HVCAL session\n");
    return -1;
  }

  hvcal_set_throughput_mode(global_session, THROUGHPUT_QUALITY);

  printf("[VIDEO KERNEL] Initialized successfully (4K H.265).\n");
  return 0;
}

int video_kernel_process(FrameBuffer *input, FrameBuffer *output) {
  if (!input || !output || !global_session)
    return -1;

  // 1. Prepare HVCAL buffer from FrameBuffer
  hvcal_buffer_t hv_buf;
  hv_buf.cpu_ptr = input->data;
  hv_buf.size = input->size;
  hv_buf.in_use = true;
  hv_buf.fd = -1; // Not using DMA-BUF in this bridge path for now

  // 2. Submit frame to hardware encoder
  if (hvcal_submit_frame(global_session, &hv_buf) != 0) {
    return -1;
  }

  // 3. Retrieve encoded bitstream
  uint8_t *bitstream_data = NULL;
  size_t bitstream_size = 0;

  if (hvcal_get_bitstream(global_session, &bitstream_data, &bitstream_size) ==
      0) {
    if (bitstream_size > 0 && output->data) {
      // Copy to output buffer (In production, we would use zero-copy or pool)
      memcpy(output->data, bitstream_data, bitstream_size);
      output->size = bitstream_size;
      output->width = global_config.width;
      output->height = global_config.height;
      output->timestamp = input->timestamp;
      output->flags |= 4; // Mark as encoded
      return 0;
    }
  }

  return -1; // No bitstream available yet (B-frame latency) or error
}

int video_kernel_finalize(void) {
  if (global_session) {
    hvcal_destroy_session(global_session);
    global_session = NULL;
  }
  printf("[VIDEO KERNEL] Finalized.\n");
  return 0;
}

// Interface struct for export
KernelInterface video_kernel_interface = {
    video_kernel_initialize, video_kernel_process, video_kernel_finalize};
