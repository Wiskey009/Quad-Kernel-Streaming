#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>


#if defined(__AVX__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define ECA_USE_AVX
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define ECA_USE_NEON
#endif

namespace audio {

template <typename T, size_t Capacity> class RingBuffer {
public:
  RingBuffer() : head_(0), tail_(0) {}
  bool push(const T *data, size_t count) noexcept;
  bool pop(T *dest, size_t count) noexcept;
  const T *data() const { return buffer_; }
  T *data() { return buffer_; }

private:
  alignas(64) T buffer_[Capacity];
  std::atomic<size_t> head_;
  std::atomic<size_t> tail_;
};

class EchoCancellerAdvanced {
public:
  EchoCancellerAdvanced(int sample_rate, int frame_size, int num_channels);
  ~EchoCancellerAdvanced();

  void process(const float *nearend, const float *farend, float *out);
  void set_adaptation(bool enable) noexcept { adaptation_.store(enable); }
  float get_erl() const noexcept { return erl_.load(); }

private:
  class DoubleTalkDetector {
  public:
    bool is_active() const noexcept { return vad_ && (ratio_ > threshold_); }
    void update(float near_level, float far_level);

  private:
    float ratio_ = 0.0f;
    bool vad_ = false;
    const float threshold_ = 0.5f;
  };

  void initialize_filters();
  void adaptive_filter(const float *farend, const float *nearend, float *out,
                       int ch);
  void simd_filter(const float *farend, const float *filter, float *out);
  void simd_update_filter(const float *farend, const float *error,
                          float *filter);

  const int sample_rate_;
  const int frame_size_;
  const int num_channels_;
  const int filter_length_;

  std::vector<std::unique_ptr<float[]>> filters_;
  RingBuffer<float, 32768> nearend_buf_;
  RingBuffer<float, 131072> farend_buf_;

  std::atomic<bool> adaptation_{true};
  std::atomic<float> erl_{0.0f};
  DoubleTalkDetector dtd_;
};

} // namespace audio
