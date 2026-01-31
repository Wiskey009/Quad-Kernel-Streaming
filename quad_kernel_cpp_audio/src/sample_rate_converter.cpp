#include "sample_rate_converter.h"
#include <algorithm>
#include <numeric>

namespace {
constexpr double PI = 3.14159265358979323846;

double kaiser_bessel(double n, double alpha) {
  double t = n / alpha;
  if (t > 1.0)
    return 0.0;
  double arg = alpha * sqrt(1.0 - t * t);
  // Simple approximation of I0 bessel function for windowing
  double numerator = 1.0;
  double term = 1.0;
  for (int i = 1; i < 20; ++i) {
    term *= (arg * arg) / (4.0 * i * i);
    numerator += term;
  }
  return numerator;
}

double sinc(double x) {
  return (std::abs(x) < 1e-9) ? 1.0 : std::sin(PI * x) / (PI * x);
}
} // namespace
namespace audio {

SampleRateConverter::SampleRateConverter(double input_rate, double output_rate,
                                         size_t num_taps, size_t num_phases,
                                         double cutoff_ratio)
    : input_rate_(input_rate), output_rate_(output_rate),
      ratio_(output_rate / input_rate),
      num_taps_((num_taps + 7) & ~7), // Align to 8 for AVX
      num_phases_(num_phases), cutoff_ratio_(cutoff_ratio),
      history_buffer_((num_taps + 7) & ~7, 0.0f), history_pos_(0),
      workspace_(new float[(num_taps + 7) & ~7]) {

  if (input_rate <= 0 || output_rate <= 0) {
    throw std::invalid_argument("Sample rates must be positive");
  }

  design_filter();
  reset();
}

void SampleRateConverter::reset() {
  std::fill(history_buffer_.begin(), history_buffer_.end(), 0.0f);
  history_pos_ = 0;
}

void SampleRateConverter::design_filter() {
  const double actual_cutoff =
      std::min(input_rate_, output_rate_) * cutoff_ratio_ * 0.5;
  const double alpha = 5.0;

  filter_bank_.resize(num_phases_);
  const size_t total_taps = num_taps_ * num_phases_;

  std::vector<float> prototype(total_taps);
  double i0_alpha = kaiser_bessel(alpha, alpha);

  for (size_t i = 0; i < total_taps; ++i) {
    double t = (static_cast<double>(i) - total_taps / 2.0) / num_phases_;
    double ideal = 2.0 * (actual_cutoff / input_rate_) *
                   sinc(2.0 * (actual_cutoff / input_rate_) * t);

    double window_t = 2.0 * static_cast<double>(i) / total_taps - 1.0;
    double window =
        kaiser_bessel(alpha * std::sqrt(1.0 - window_t * window_t), alpha) /
        i0_alpha;

    prototype[i] = static_cast<float>(ideal * window);
  }

  for (size_t phase = 0; phase < num_phases_; ++phase) {
    auto &fp = filter_bank_[phase];
    fp.coefficients.resize(num_taps_);
    fp.offset = 0; // Simplified for this implementation

    for (size_t tap = 0; tap < num_taps_; ++tap) {
      size_t idx = phase + tap * num_phases_;
      fp.coefficients[tap] = prototype[idx];
    }
  }
}

void SampleRateConverter::process(const float *input, float *output,
                                  size_t input_length) {
  if (input_length == 0)
    return;

  double phase_step = input_rate_ / output_rate_;
  double current_phase = 0.0;
  size_t output_index = 0;

  for (size_t i = 0; i < input_length; ++i) {
    // Simple implementation: this needs a more robust history management for
    // production For now, let's assume we are processing blocks and history is
    // handled outside OR we implement the sliding window properly.

    // Proper sliding window:
    history_buffer_[history_pos_] = input[i];

    while (current_phase < 1.0) {
      size_t phase_idx = static_cast<size_t>(current_phase * num_phases_);
      if (phase_idx >= num_phases_)
        phase_idx = num_phases_ - 1;

#if defined(USE_AVX)
      compute_simd_avx(history_buffer_.data(), output + output_index,
                       phase_idx);
#else
      compute_single_output(history_buffer_.data(), output + output_index,
                            phase_idx);
#endif
      output_index++;
      current_phase += phase_step;
    }
    current_phase -= 1.0;
    history_pos_ = (history_pos_ + 1) % num_taps_;
  }
}

void SampleRateConverter::compute_single_output(const float *input,
                                                float *output,
                                                size_t phase_index) {
  const auto &phase = filter_bank_[phase_index];
  float sum = 0.0f;
  for (size_t i = 0; i < num_taps_; ++i) {
    size_t idx = (history_pos_ + num_taps_ - i) % num_taps_;
    sum += input[idx] * phase.coefficients[i];
  }
  *output = sum;
}

#if defined(USE_AVX)
void SampleRateConverter::compute_simd_avx(const float *input, float *output,
                                           size_t phase_index) {
  const auto &phase = filter_bank_[phase_index];
  __m256 sum = _mm256_setzero_ps();

  // This requires history_buffer to be linear for simplicity or handled
  // correctly. For AVX to be efficient, the data must be aligned and
  // contiguous. Simplifying here:
  for (size_t i = 0; i < num_taps_; i += 8) {
    // This is still a bit complex due to circular buffer.
    // In a real implementation, we'd double the history buffer to avoid modulo.
    float temp_input[8];
    for (size_t j = 0; j < 8; ++j)
      temp_input[j] = input[(history_pos_ + num_taps_ - (i + j)) % num_taps_];

    __m256 i_vec = _mm256_loadu_ps(temp_input);
    __m256 c_vec = _mm256_loadu_ps(&phase.coefficients[i]);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(i_vec, c_vec));
  }

  alignas(32) float res[8];
  _mm256_store_ps(res, sum);
  *output =
      res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7];
}
#endif
} // namespace audio
