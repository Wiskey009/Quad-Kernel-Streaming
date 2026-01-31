#pragma once
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>


#if defined(__AVX__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define DSP_USE_AVX
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define DSP_USE_NEON
#endif

namespace dsp {

class NoiseSuppressor {
public:
  struct Config {
    float sample_rate = 48000.0f;
    int frame_size = 512;
    int overlap_factor = 2;
    float noise_gate_db = -30.0f;
  };

  explicit NoiseSuppressor(const Config &cfg);
  ~NoiseSuppressor();

  void process(float *input, float *output, int num_channels, int num_frames);

  using NoiseEstimator = float (*)(const float *spectrum, int bin_size,
                                   void *context);
  void set_noise_estimator(NoiseEstimator estimator, void *context = nullptr);

  NoiseSuppressor(const NoiseSuppressor &) = delete;
  NoiseSuppressor &operator=(const NoiseSuppressor &) = delete;
  NoiseSuppressor(NoiseSuppressor &&) noexcept;
  NoiseSuppressor &operator=(NoiseSuppressor &&) noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace dsp
