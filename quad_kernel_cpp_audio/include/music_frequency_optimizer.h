#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>


#if defined(__AVX__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define MFO_USE_AVX
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define MFO_USE_NEON
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace dsp {

struct FrequencyProfile {
  float low_cutoff = 200.0f;
  float high_cutoff = 16000.0f;
  float spectral_tilt = -1.2f;
};

class MusicFrequencyOptimizer {
public:
  MusicFrequencyOptimizer(int sample_rate, int frame_size);
  ~MusicFrequencyOptimizer();

  void configure(const FrequencyProfile &profile) noexcept;
  void process(const float *input, float *output) noexcept;

private:
  void simd_fft_analyze() noexcept;
  void apply_spectral_tilt(float *spectrum) noexcept;

  const int sample_rate_;
  const int frame_size_;
  FrequencyProfile profile_;

  struct AlignedDeleter {
    void operator()(float *p) const;
  };
  std::unique_ptr<float[], AlignedDeleter> fft_buffer_;
  std::atomic<bool> config_updated_{false};
};

} // namespace dsp
