#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <immintrin.h>
#include <memory>
#include <stdexcept>
#include <variant>
#include <vector>


#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace audio {

enum class Format { PCM, WAV, FLAC, Opus, AAC };
enum class Endianness { Little, Big };

struct AudioParams {
  Format format;
  uint32_t sample_rate;
  uint16_t channels;
  uint16_t bits_per_sample;
  Endianness endianness = Endianness::Little;
};

class IBufferedResource {
public:
  virtual ~IBufferedResource() = default;
  virtual void *acquire_buffer(size_t bytes) = 0;
  virtual void release_buffer(void *buffer) = 0;
};

class LockFreeBufferPool : public IBufferedResource {
  struct alignas(64) BufferNode {
    std::atomic<BufferNode *> next;
    uint8_t data[];
  };

  std::atomic<BufferNode *> head_;
  const size_t buffer_size_;
  const uint32_t max_buffers_;

public:
  explicit LockFreeBufferPool(size_t buffer_size, uint32_t count);
  ~LockFreeBufferPool() override;

  void *acquire_buffer(size_t bytes) override;
  void release_buffer(void *buffer) override;
};

class AudioFormatConverter {
protected:
  AudioParams src_params_;
  AudioParams dst_params_;
  IBufferedResource &buffer_pool_;

  static bool is_avx2_supported();

public:
  AudioFormatConverter(AudioParams src, AudioParams dst,
                       IBufferedResource &pool)
      : src_params_(std::move(src)), dst_params_(std::move(dst)),
        buffer_pool_(pool) {}
  virtual ~AudioFormatConverter() = default;

  virtual size_t convert(const void *src, size_t src_bytes, void *dst,
                         size_t dst_capacity) = 0;
};

class PCMConverter final : public AudioFormatConverter {
public:
  using AudioFormatConverter::AudioFormatConverter;

  size_t convert(const void *src, size_t src_bytes, void *dst,
                 size_t dst_capacity) override;

private:
  size_t generic_convert(const void *src, size_t src_bytes, void *dst,
                         size_t dst_capacity);

  void convert_16_to_32_avx2(const int16_t *src, int32_t *dst, size_t samples);
};

} // namespace audio
