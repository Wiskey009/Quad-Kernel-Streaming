#include "hvcal_core.h"
#include "nvenc_manager.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct hvcal_session {
  uint32_t device_id;
  hvcal_vendor_t vendor;
  hvcal_codec_t codec;
  hvcal_config_t config;

  // Vendor specific managers
  void *manager;
};

int hvcal_initialize(bool enable_debug) {
  if (enable_debug) {
    printf("[HVCAL] Initializing system in debug mode\n");
  }
  return 0;
}

int hvcal_enumerate_devices(hvcal_device_info_t *devices,
                            uint32_t max_devices) {
  if (!devices || max_devices == 0)
    return 0;

  // Mock discovery
  devices[0].device_id = 0;
  devices[0].vendor = HVCAL_VENDOR_NVIDIA;
  strcpy(devices[0].name, "NVIDIA GeForce RTX 4090");
  devices[0].total_vram = 24LL * 1024 * 1024 * 1024;
  devices[0].max_resolution_w = 8192;
  devices[0].max_resolution_h = 8192;

  return 1;
}

hvcal_session_t *hvcal_create_session(uint32_t device_id, hvcal_codec_t codec,
                                      const hvcal_config_t *config) {
  hvcal_session_t *session = (hvcal_session_t *)malloc(sizeof(hvcal_session_t));
  if (!session)
    return NULL;

  session->device_id = device_id;
  session->codec = codec;
  session->config = *config;
  session->manager = NULL;

  // For now, default to NVIDIA if device 0
  if (device_id == 0) {
    session->vendor = HVCAL_VENDOR_NVIDIA;
    nvenc_manager_config_t nv_cfg = {.width = config->width,
                                     .height = config->height,
                                     .codec = codec,
                                     .fps = config->fps,
                                     .bitrate = config->bitrate};
    session->manager = nvenc_manager_create(device_id, &nv_cfg);

    // Safety check: If manager failed to create, cleanup session
    if (!session->manager) {
      free(session);
      return NULL;
    }
  } else {
    free(session);
    return NULL;
  }

  return session;
}

int hvcal_destroy_session(hvcal_session_t *session) {
  if (!session)
    return -1;

  if (session->vendor == HVCAL_VENDOR_NVIDIA) {
    nvenc_manager_destroy((nvenc_manager_ctx_t *)session->manager);
  }

  free(session);
  return 0;
}

int hvcal_submit_frame(hvcal_session_t *session, const hvcal_buffer_t *frame) {
  if (!session || !frame)
    return -1;

  if (session->vendor == HVCAL_VENDOR_NVIDIA) {
    return nvenc_manager_submit_frame((nvenc_manager_ctx_t *)session->manager,
                                      frame);
  }

  return -2;
}

int hvcal_get_bitstream(hvcal_session_t *session, uint8_t **data,
                        size_t *size) {
  if (!session || !data || !size)
    return -1;

  if (session->vendor == HVCAL_VENDOR_NVIDIA) {
    return nvenc_manager_get_bitstream((nvenc_manager_ctx_t *)session->manager,
                                       data, size);
  }

  return -2;
}

void hvcal_set_throughput_mode(hvcal_session_t *session, uint8_t mode) {
  printf("[HVCAL] Setting throughput mode to %u\n", mode);
}
