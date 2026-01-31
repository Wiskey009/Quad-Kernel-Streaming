#ifndef HVCAL_CORE_H
#define HVCAL_CORE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#define HVCAL_VERSION_MAJOR 1
#define HVCAL_VERSION_MINOR 0
#define MAX_DEVICES 16
#define MAX_STREAMS 32
#define MAX_SURFACES 64

typedef enum {
    HVCAL_CODEC_H264 = 0x1,
    HVCAL_CODEC_H265 = 0x2,
    HVCAL_CODEC_AV1  = 0x4,
    HVCAL_CODEC_VP9  = 0x8
} hvcal_codec_t;

typedef enum {
    HVCAL_VENDOR_NVIDIA = 0x1,
    HVCAL_VENDOR_INTEL  = 0x2,
    HVCAL_VENDOR_AMD    = 0x4,
    HVCAL_VENDOR_GENERIC = 0x8
} hvcal_vendor_t;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t fps;
    uint32_t bitrate;
    uint16_t gop_size;
    uint8_t bit_depth;
    bool hdr;
} hvcal_config_t;

typedef struct {
    int fd;                 // DMA-BUF file descriptor
    void* cpu_ptr;          // CPU mapping (if needed)
    uint64_t dma_address;   // Physical address
    size_t size;
    bool in_use;
} hvcal_buffer_t;

typedef struct {
    uint32_t device_id;
    hvcal_vendor_t vendor;
    char name[256];
    uint64_t total_vram;
    uint32_t max_resolution_w;
    uint32_t max_resolution_h;
} hvcal_device_info_t;

// Opaque session handle
typedef struct hvcal_session hvcal_session_t;

// Initialization
int hvcal_initialize(bool enable_debug);

// Device Management
int hvcal_enumerate_devices(hvcal_device_info_t* devices, uint32_t max_devices);

// Session Control
hvcal_session_t* hvcal_create_session(uint32_t device_id, hvcal_codec_t codec, const hvcal_config_t* config);
int hvcal_destroy_session(hvcal_session_t* session);

// Frame Processing
int hvcal_submit_frame(hvcal_session_t* session, const hvcal_buffer_t* frame);
int hvcal_get_bitstream(hvcal_session_t* session, uint8_t** data, size_t* size);

// Performance Tuning
#define THROUGHPUT_ULTRA 0
#define THROUGHPUT_BALANCED 1
#define THROUGHPUT_QUALITY 2
void hvcal_set_throughput_mode(hvcal_session_t* session, uint8_t mode);

#endif // HVCAL_CORE_H
