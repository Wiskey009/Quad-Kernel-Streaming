#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <type_traits>


#if defined(__AVX__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define RINGBUFFER_USE_AVX
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define RINGBUFFER_USE_NEON
#endif

namespace audio {

template <typename T, size_t Capacity> class AudioRingBuffer {
public:
  AudioRingBuffer() : head_(0), tail_(0) {
    static_assert(Capacity > 0 && (Capacity & (Capacity - 1)) == 0,
                  "Capacity must be power of two");
    static_assert(std::is_trivial_v<T>, "T must be trivial type");
  }

  // Disallow copying
  AudioRingBuffer(const AudioRingBuffer &) = delete;
  AudioRingBuffer &operator=(const AudioRingBuffer &) = delete;

  // Move semantics
  AudioRingBuffer(AudioRingBuffer &&) = default;
  AudioRingBuffer &operator=(AudioRingBuffer &&) = default;

  size_t capacity() const noexcept { return Capacity; }

  struct Chunk {
    T *data;
    size_t size;
  };

  // Producer side
  Chunk prepare_write(size_t requested) noexcept {
    const size_t wr = tail_.load(std::memory_order_relaxed);
    const size_t rd = head_.load(std::memory_order_acquire);
    const size_t avail = Capacity - (wr - rd);

    if (avail < requested) {
      if (avail == 0)
        return {nullptr, 0};
      requested = avail;
    }

    const size_t idx = wr & (Capacity - 1);
    const size_t contig = Capacity - idx;
    return {buffer_.get() + idx, std::min(requested, contig)};
  }

  void commit_write(size_t written) noexcept {
    tail_.fetch_add(written, std::memory_order_release);
  }

  // Consumer side
  Chunk prepare_read(size_t requested) noexcept {
    const size_t rd = head_.load(std::memory_order_relaxed);
    const size_t wr = tail_.load(std::memory_order_acquire);
    const size_t avail = wr - rd;

    if (avail < requested) {
      if (avail == 0)
        return {nullptr, 0};
      requested = avail;
    }

    const size_t idx = rd & (Capacity - 1);
    const size_t contig = Capacity - idx;
    return {buffer_.get() + idx, std::min(requested, contig)};
  }

  void commit_read(size_t read) noexcept {
    head_.fetch_add(read, std::memory_order_release);
  }

#ifdef RINGBUFFER_USE_AVX
  size_t write_aligned(const __m256 *src, size_t count) noexcept {
    const size_t elements_per_vec = 8;
    const size_t requested = count * elements_per_vec;

    Chunk chunk = prepare_write(requested);
    if (!chunk.data || chunk.size < requested)
      return 0;

    for (size_t i = 0; i < count; ++i) {
      _mm256_storeu_ps(
          reinterpret_cast<float *>(chunk.data + i * elements_per_vec), src[i]);
    }

    commit_write(requested);
    return count;
  }
#endif

private:
  alignas(64) std::unique_ptr<T[]> buffer_{new T[Capacity]};
  std::atomic<size_t> head_{0};
  std::atomic<size_t> tail_{0};
};

} // namespace audio
