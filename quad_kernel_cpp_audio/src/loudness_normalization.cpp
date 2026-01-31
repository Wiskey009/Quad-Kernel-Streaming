#include "loudness_normalization.h"
#include <algorithm>
#include <cmath>
#include <numeric>


#if defined(__AVX__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define LN_USE_AVX
#endif

namespace {
// ITU-R BS.1770-4 Coefficients
constexpr double HP_COEFFS[5] = {1.0, -1.990047, 0.990072, -1.995113, 0.995127};
constexpr double SHF1_COEFFS[5] = {1.0, -1.690293, 0.715112, -1.719469,
                                   0.731726};
constexpr double SHF2_COEFFS[5] = {1.0, -1.934905, 0.935556, -1.942779,
                                   0.943685};

inline void apply_filter(const double coeffs[5], double *state, double input,
                         double &output) {
  output = coeffs[0] * input + coeffs[1] * state[0] + coeffs[2] * state[1];
  output -= coeffs[3] * state[2] + coeffs[4] * state[3];
  state[1] = state[0];
  state[0] = input;
  state[3] = state[2];
  state[2] = output;
}
} // namespace

struct LoudnessNormalizer::FilterState {
  double hp_out[4]{0};
  double shf1_out[4]{0};
  double shf2_out[4]{0};
  std::vector<float> block_energy;
  double total_energy = 0.0;
  size_t sample_count = 0;
  size_t sample_rate;
  static constexpr size_t BLOCK_SIZE = 1024;

  FilterState(size_t sr) : sample_rate(sr) { block_energy.reserve(BLOCK_SIZE); }
};

LoudnessNormalizer::LoudnessNormalizer(size_t sample_rate)
    : state_(std::make_unique<FilterState>(sample_rate)) {}

LoudnessNormalizer::~LoudnessNormalizer() = default;

void LoudnessNormalizer::process(const float *input, size_t num_samples) {
  double filtered = 0.0;
  for (size_t i = 0; i < num_samples; ++i) {
    apply_filter(HP_COEFFS, state_->hp_out, input[i], filtered);
    apply_filter(SHF1_COEFFS, state_->shf1_out, filtered, filtered);
    apply_filter(SHF2_COEFFS, state_->shf2_out, filtered, filtered);

    state_->block_energy.push_back(static_cast<float>(filtered));
    if (state_->block_energy.size() >= FilterState::BLOCK_SIZE) {
      double block_sum = 0.0;
      size_t k = 0;
#ifdef LN_USE_AVX
      __m256 v_sum = _mm256_setzero_ps();
      for (; k + 7 < state_->block_energy.size(); k += 8) {
        __m256 x = _mm256_loadu_ps(&state_->block_energy[k]);
        v_sum = _mm256_add_ps(v_sum, _mm256_mul_ps(x, x));
      }
      float tmp[8];
      _mm256_storeu_ps(tmp, v_sum);
      for (float t : tmp)
        block_sum += t;
#endif
      for (; k < state_->block_energy.size(); ++k) {
        float val = state_->block_energy[k];
        block_sum += val * val;
      }
      state_->total_energy += block_sum;
      state_->sample_count += state_->block_energy.size();
      state_->block_energy.clear();
    }
  }
}

double LoudnessNormalizer::get_loudness_lkfs() const {
  if (state_->sample_count == 0)
    return -70.0;
  const double energy = state_->total_energy / state_->sample_count;
  if (energy <= 1e-7)
    return -70.0;
  return -0.691 + 10.0 * log10(energy);
}

void LoudnessNormalizer::reset() {
  std::fill_n(state_->hp_out, 4, 0.0);
  std::fill_n(state_->shf1_out, 4, 0.0);
  std::fill_n(state_->shf2_out, 4, 0.0);
  state_->total_energy = 0.0;
  state_->sample_count = 0;
  state_->block_energy.clear();
}
