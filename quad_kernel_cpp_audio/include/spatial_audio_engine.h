#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>


#if defined(__AVX__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define SAE_USE_AVX
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define SAE_USE_NEON
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace audio {

struct SpatialParams {
  float azimuth;
  float elevation;
  float distance;
};

struct AudioBlock {
  const float *data;
  size_t samples;
};

template <typename T, size_t Capacity> class LockFreeQueue {
public:
  bool push(const T &item) {
    size_t h = head_.load(std::memory_order_relaxed);
    size_t next = (h + 1) % Capacity;
    if (next == tail_.load(std::memory_order_acquire))
      return false;
    buffer_[h] = item;
    head_.store(next, std::memory_order_release);
    return true;
  }
  bool pop(T &item) {
    size_t t = tail_.load(std::memory_order_relaxed);
    if (t == head_.load(std::memory_order_acquire))
      return false;
    item = buffer_[t];
    tail_.store((t + 1) % Capacity, std::memory_order_release);
    return true;
  }
  size_t size() const {
    size_t h = head_.load(std::memory_order_acquire);
    size_t t = tail_.load(std::memory_order_acquire);
    return (h >= t) ? (h - t) : (Capacity - t + h);
  }

private:
  T buffer_[Capacity];
  std::atomic<size_t> head_{0};
  std::atomic<size_t> tail_{0};
};

class HRTFDatabase {
public:
  HRTFDatabase(size_t max_frames, size_t fft_size);
  void load_frame(size_t index, const float *left, const float *right);
  void get_interpolated_frame(float az, float el, float *left,
                              float *right) const;
  size_t get_fft_size() const { return fft_size_; }

private:
  std::vector<float> left_ir_, right_ir_;
  size_t max_frames_, fft_size_;
};

class SpatialAudioSource {
public:
  SpatialAudioSource(size_t buffer_frames, size_t fft_size);
  void process_block(const AudioBlock &block, const HRTFDatabase &hrtf,
                     const SpatialParams &params, float *out_l, float *out_r);

private:
  void process_convolution(const HRTFDatabase &hrtf,
                           const SpatialParams &params);
  LockFreeQueue<float, 4096> input_queue_;
  std::vector<float> prev_l_out_, prev_r_out_;
  size_t fft_size_;
};

class SpatialAudioEngine {
public:
  SpatialAudioEngine(size_t num_sources, size_t block_size, size_t fft_size);
  void process(const AudioBlock *inputs, const SpatialParams *params,
               size_t num_blocks, float *output_l, float *output_r);
  void set_hrtf_database(std::unique_ptr<HRTFDatabase> db);

private:
  std::vector<std::unique_ptr<SpatialAudioSource>> sources_;
  std::unique_ptr<HRTFDatabase> hrtf_db_;
  std::vector<float> accum_l_, accum_r_;
  size_t block_size_, fft_size_;
};

} // namespace audio
