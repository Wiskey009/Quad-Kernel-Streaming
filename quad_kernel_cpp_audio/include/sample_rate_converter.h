#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(__AVX__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define USE_AVX
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define USE_NEON
#endif

namespace audio {
class SampleRateConverter {
public:
  SampleRateConverter(double input_rate, double output_rate,
                      size_t num_taps = 64, size_t num_phases = 32,
                      double cutoff_ratio = 0.95);

  void process(const float *input, float *output, size_t input_length);
  void reset();

private:
  struct FilterPhase {
    size_t offset;
    std::vector<float> coefficients;
  };

  void design_filter();
  void compute_single_output(const float *input, float *output,
                             size_t phase_index);

#if defined(USE_AVX)
  void compute_simd_avx(const float *input, float *output, size_t phase_index);
#elif defined(USE_NEON)
  void compute_simd_neon(const float *input, float *output, size_t phase_index);
#endif

  const double input_rate_;
  const double output_rate_;
  const double ratio_;
  const size_t num_taps_;
  const size_t num_phases_;
  const double cutoff_ratio_;

  std::vector<FilterPhase> filter_bank_;
  std::vector<float> history_buffer_;
  size_t history_pos_;
  std::unique_ptr<float[]> workspace_;
};
} // namespace audio
