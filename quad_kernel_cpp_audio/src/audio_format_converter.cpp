#include "audio_format_converter.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>


#ifdef _WIN32
#include <malloc.h>
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_free(ptr) free(ptr)
#endif

namespace audio {

LockFreeBufferPool::LockFreeBufferPool(size_t buffer_size, uint32_t count)
    : head_(nullptr), buffer_size_(buffer_size), max_buffers_(count) {
  for (uint32_t i = 0; i < count; ++i) {
    auto *node = reinterpret_cast<BufferNode *>(
        aligned_alloc(64, sizeof(BufferNode) + buffer_size));
    node->next.store(head_.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
    head_.store(node, std::memory_order_relaxed);
  }
}

LockFreeBufferPool::~LockFreeBufferPool() {
  BufferNode *node = head_.load(std::memory_order_relaxed);
  while (node) {
    BufferNode *next = node->next.load(std::memory_order_relaxed);
    aligned_free(node);
    node = next;
  }
}

void *LockFreeBufferPool::acquire_buffer(size_t bytes) {
  if (bytes > buffer_size_)
    throw std::bad_alloc{};
  BufferNode *old_head = head_.load(std::memory_order_acquire);
  while (old_head &&
         !head_.compare_exchange_weak(
             old_head, old_head->next.load(std::memory_order_relaxed),
             std::memory_order_acq_rel)) {
    // Retry
  }
  return old_head ? old_head->data : nullptr;
}

void LockFreeBufferPool::release_buffer(void *buffer) {
  if (!buffer)
    return;
  auto *node = reinterpret_cast<BufferNode *>(
      reinterpret_cast<uint8_t *>(buffer) - offsetof(BufferNode, data));
  BufferNode *old_head = head_.load(std::memory_order_acquire);
  do {
    node->next.store(old_head, std::memory_order_relaxed);
  } while (
      !head_.compare_exchange_weak(old_head, node, std::memory_order_acq_rel));
}

bool AudioFormatConverter::is_avx2_supported() {
#ifdef _MSC_VER
  int cpu_info[4];
  __cpuid(cpu_info, 7);
  return (cpu_info[1] & (1 << 5)) != 0; // EBX bit 5 is AVX2
#else
  uint32_t eax, ebx, ecx, edx;
  if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    return (ebx & (1 << 5)) != 0;
  }
  return false;
#endif
}

size_t PCMConverter::convert(const void *src, size_t src_bytes, void *dst,
                             size_t dst_capacity) {
  const size_t src_sample_size = src_params_.bits_per_sample / 8;
  const size_t dst_sample_size = dst_params_.bits_per_sample / 8;
  const size_t samples = src_bytes / src_sample_size;
  const size_t required_bytes = samples * dst_sample_size;

  if (dst_capacity < required_bytes)
    throw std::length_error("Insufficient dst capacity");

  if (is_avx2_supported() && src_params_.bits_per_sample == 16 &&
      dst_params_.bits_per_sample == 32) {
    convert_16_to_32_avx2(static_cast<const int16_t *>(src),
                          static_cast<int32_t *>(dst), samples);
    return required_bytes;
  }

  return generic_convert(src, src_bytes, dst, dst_capacity);
}

size_t PCMConverter::generic_convert(const void *src, size_t src_bytes,
                                     void *dst, size_t dst_capacity) {
  const size_t samples = src_bytes / (src_params_.bits_per_sample / 8);
  const size_t dst_sample_size = dst_params_.bits_per_sample / 8;

  if (src_params_.bits_per_sample == 16 && dst_params_.bits_per_sample == 32) {
    const int16_t *s = static_cast<const int16_t *>(src);
    int32_t *d = static_cast<int32_t *>(dst);
    for (size_t i = 0; i < samples; ++i) {
      d[i] = static_cast<int32_t>(s[i]) << 16;
    }
    return samples * 4;
  }
  // Add more generic conversions as needed...
  return 0;
}

void PCMConverter::convert_16_to_32_avx2(const int16_t *src, int32_t *dst,
                                         size_t samples) {
  size_t i = 0;
  for (; i + 7 < samples; i += 8) {
    __m128i v16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i));
    __m256i v32 = _mm256_cvtepi16_epi32(v16);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i), v32);
  }
  for (; i < samples; ++i) {
    dst[i] = static_cast<int32_t>(src[i]) << 16;
  }
}

} // namespace audio
