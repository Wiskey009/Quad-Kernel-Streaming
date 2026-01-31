#include "music_frequency_optimizer.h"
#include <algorithm>
#include <cmath>
#include <cstring>


#ifdef _WIN32
#include <malloc.h>
#define MFO_ALIGNED_ALLOC(size) _aligned_malloc(size, 32)
#define MFO_ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#include <stdlib.h>
static inline void *MFO_ALIGNED_ALLOC(size_t size) {
  void *ptr = nullptr;
  if (posix_memalign(&ptr, 32, size) != 0)
    return nullptr;
  return ptr;
}
#define MFO_ALIGNED_FREE(ptr) free(ptr)
#endif

namespace dsp {

void MusicFrequencyOptimizer::AlignedDeleter::operator()(float *p) const {
  if (p)
    MFO_ALIGNED_FREE(p);
}

MusicFrequencyOptimizer::MusicFrequencyOptimizer(int sample_rate,
                                                 int frame_size)
    : sample_rate_(sample_rate), frame_size_(frame_size) {
  fft_buffer_.reset(
      static_cast<float *>(MFO_ALIGNED_ALLOC(frame_size * 2 * sizeof(float))));
}

MusicFrequencyOptimizer::~MusicFrequencyOptimizer() = default;

void MusicFrequencyOptimizer::configure(
    const FrequencyProfile &profile) noexcept {
  profile_ = profile;
  config_updated_.store(true, std::memory_order_release);
}

void MusicFrequencyOptimizer::process(const float *input,
                                      float *output) noexcept {
  std::memcpy(fft_buffer_.get(), input, frame_size_ * sizeof(float));
  simd_fft_analyze();
  float *spectrum = fft_buffer_.get() + frame_size_;
  apply_spectral_tilt(spectrum);
  std::memcpy(output, fft_buffer_.get(), frame_size_ * sizeof(float));
}

void MusicFrequencyOptimizer::apply_spectral_tilt(float *spectrum) noexcept {
  const int num_bins = frame_size_ / 2;
  const float tilt_gain = config_updated_.exchange(false)
                              ? profile_.spectral_tilt
                              : profile_.spectral_tilt;

  int i = 0;
#ifdef MFO_USE_AVX
  __m256 v_tilt = _mm256_set1_ps(tilt_gain);
  float inv_bins = 1.0f / static_cast<float>(num_bins);
  for (; i + 7 < num_bins; i += 8) {
    __m256 v_idx =
        _mm256_setr_ps((float)i, (float)i + 1, (float)i + 2, (float)i + 3,
                       (float)i + 4, (float)i + 5, (float)i + 6, (float)i + 7);
    __m256 v_weight =
        _mm256_mul_ps(_mm256_mul_ps(v_idx, _mm256_set1_ps(inv_bins)), v_tilt);
    __m256 v_bin = _mm256_load_ps(&spectrum[i]);
    _mm256_store_ps(&spectrum[i], _mm256_add_ps(v_bin, v_weight));
  }
#endif
  for (; i < num_bins; ++i) {
    spectrum[i] += (static_cast<float>(i) / num_bins) * tilt_gain;
  }
}

void MusicFrequencyOptimizer::simd_fft_analyze() noexcept {
  for (int i = 0; i < frame_size_; ++i) {
    fft_buffer_.get()[frame_size_ + i] =
        fft_buffer_.get()[i] *
        sinf(static_cast<float>(2.0 * M_PI * i / frame_size_));
  }
}

} // namespace dsp
