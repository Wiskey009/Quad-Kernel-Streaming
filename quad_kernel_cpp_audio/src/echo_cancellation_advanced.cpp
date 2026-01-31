#include "echo_cancellation_advanced.h"
#include <algorithm>
#include <cmath>
#include <cstring>


namespace audio {

template <typename T, size_t Capacity>
bool RingBuffer<T, Capacity>::push(const T *data, size_t count) noexcept {
  const size_t h = head_.load(std::memory_order_relaxed);
  const size_t t = tail_.load(std::memory_order_acquire);
  if (Capacity - (h - t) < count)
    return false;

  size_t offset = h % Capacity;
  size_t first_part = std::min(count, Capacity - offset);
  std::memcpy(buffer_ + offset, data, first_part * sizeof(T));
  if (count > first_part) {
    std::memcpy(buffer_, data + first_part, (count - first_part) * sizeof(T));
  }
  head_.store(h + count, std::memory_order_release);
  return true;
}

template <typename T, size_t Capacity>
bool RingBuffer<T, Capacity>::pop(T *dest, size_t count) noexcept {
  const size_t h = head_.load(std::memory_order_acquire);
  const size_t t = tail_.load(std::memory_order_relaxed);
  if (h - t < count)
    return false;

  size_t offset = t % Capacity;
  size_t first_part = std::min(count, Capacity - offset);
  std::memcpy(dest, buffer_ + offset, first_part * sizeof(T));
  if (count > first_part) {
    std::memcpy(dest + first_part, buffer_, (count - first_part) * sizeof(T));
  }
  tail_.store(t + count, std::memory_order_release);
  return true;
}

void EchoCancellerAdvanced::DoubleTalkDetector::update(float near_level,
                                                       float far_level) {
  ratio_ = near_level / (far_level + 1e-10f);
  vad_ = (near_level > 0.03f); // -30 dB approx
}

EchoCancellerAdvanced::EchoCancellerAdvanced(int sample_rate, int frame_size,
                                             int num_channels)
    : sample_rate_(sample_rate), frame_size_(frame_size),
      num_channels_(num_channels),
      filter_length_(std::min(1024, sample_rate * 32 / 1000)) {
  initialize_filters();
}

EchoCancellerAdvanced::~EchoCancellerAdvanced() = default;

void EchoCancellerAdvanced::initialize_filters() {
  filters_.resize(num_channels_);
  for (auto &f : filters_) {
    f = std::make_unique<float[]>(filter_length_);
    std::fill_n(f.get(), filter_length_, 0.0f);
  }
}

void EchoCancellerAdvanced::process(const float *nearend, const float *farend,
                                    float *out) {
  nearend_buf_.push(nearend, frame_size_ * num_channels_);
  farend_buf_.push(farend, frame_size_ * num_channels_);

  for (int ch = 0; ch < num_channels_; ++ch) {
    float *nearend_ch = nearend_buf_.data() + ch * frame_size_;
    float *farend_ch = farend_buf_.data() + ch * filter_length_;
    adaptive_filter(farend_ch, nearend_ch, out + ch * frame_size_, ch);
  }
}

void EchoCancellerAdvanced::adaptive_filter(const float *farend,
                                            const float *nearend, float *out,
                                            int ch) {
  std::vector<float> echo_estimate(frame_size_, 0.0f);
  simd_filter(farend, filters_[ch].get(), echo_estimate.data());

  for (int i = 0; i < frame_size_; ++i) {
    out[i] = nearend[i] - echo_estimate[i];
  }

  if (adaptation_.load(std::memory_order_relaxed) && !dtd_.is_active()) {
    simd_update_filter(farend, out, filters_[ch].get());
  }
}

void EchoCancellerAdvanced::simd_filter(const float *farend,
                                        const float *filter, float *out) {
  for (int i = 0; i < frame_size_; ++i) {
    float sum = 0.0f;
    int j = 0;
#ifdef ECA_USE_AVX
    __m256 v_sum = _mm256_setzero_ps();
    for (; j + 7 < filter_length_; j += 8) {
      __m256 v_f = _mm256_loadu_ps(&filter[j]);
      __m256 v_x = _mm256_loadu_ps(&farend[i + j]);
      v_sum = _mm256_add_ps(v_sum, _mm256_mul_ps(v_f, v_x));
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, v_sum);
    for (float t : tmp)
      sum += t;
#endif
    for (; j < filter_length_; ++j) {
      sum += filter[j] * farend[i + j];
    }
    out[i] = sum;
  }
}

void EchoCancellerAdvanced::simd_update_filter(const float *farend,
                                               const float *error,
                                               float *filter) {
  const float mu = 0.01f;
  for (int i = 0; i < filter_length_; ++i) {
    filter[i] += mu * farend[i] * error[0]; // Simplified update
  }
}

} // namespace audio
