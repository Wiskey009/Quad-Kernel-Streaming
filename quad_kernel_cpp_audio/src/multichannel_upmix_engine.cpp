#include "multichannel_upmix_engine.h"
#include <algorithm>
#include <cmath>
#include <cstring>


#if defined(__AVX__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define MUE_USE_AVX
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define MUE_USE_NEON
#endif

namespace {
constexpr float FL_COEFF[2] = {0.8f, 0.6f};
constexpr float FR_COEFF[2] = {0.6f, 0.8f};
constexpr float C_COEFF = 0.707f;
constexpr float LFE_COEFF = 0.3f;
} // namespace

struct MultichannelUpmixEngine::Impl {
  const size_t output_channels;
  const float sample_rate;
  std::vector<float> delay_buffer;
  size_t write_index = 0;

  Impl(size_t ch, float sr) : output_channels(ch), sample_rate(sr) {}

  void matrix_upmix(const float *in, float *out, size_t frames) {
    for (size_t i = 0; i < frames; ++i) {
      float l = in[i * 2];
      float r = in[i * 2 + 1];
      float *dest = &out[i * output_channels];

      dest[0] = l * FL_COEFF[0] + r * FL_COEFF[1]; // L
      dest[1] = l * FR_COEFF[0] + r * FR_COEFF[1]; // R
      dest[2] = (l + r) * C_COEFF;                 // C
      dest[3] = (l + r) * LFE_COEFF;               // LFE

      if (output_channels >= 6) {
        dest[4] = l * 0.7f; // SL
        dest[5] = r * 0.7f; // SR
      }
      if (output_channels >= 8) {
        dest[6] = l * 0.5f; // RL
        dest[7] = r * 0.5f; // RR
      }
    }
  }

  void hrtf_upmix(const float *in, float *out, size_t frames) {
    // Mock HRTF: add simple 10ms delay to rear channels
    const size_t delay_samples = static_cast<size_t>(sample_rate * 0.010f);
    if (delay_buffer.size() < delay_samples + frames)
      delay_buffer.resize(delay_samples + frames, 0.0f);

    for (size_t i = 0; i < frames; ++i) {
      float l = in[i * 2];
      float r = in[i * 2 + 1];
      float *dest = &out[i * output_channels];

      // Standard Matrix for Front
      dest[0] = l;
      dest[1] = r;
      dest[2] = (l + r) * 0.707f;
      dest[3] = (l + r) * 0.2f;

      // Delayed Surround
      if (output_channels >= 6) {
        size_t r_idx = (write_index + i) % delay_buffer.size();
        dest[4] = delay_buffer[r_idx] * 0.5f;
        dest[5] = delay_buffer[r_idx] * 0.5f;
        delay_buffer[r_idx] = (l + r) * 0.5f;
      }
    }
    write_index = (write_index + frames) % delay_buffer.size();
  }
};

MultichannelUpmixEngine::MultichannelUpmixEngine(size_t output_channels,
                                                 float sample_rate,
                                                 UpmixMethod method)
    : _impl(std::make_unique<Impl>(output_channels, sample_rate)),
      _current_method(method) {}

MultichannelUpmixEngine::~MultichannelUpmixEngine() = default;

void MultichannelUpmixEngine::process(const float *stereo_in,
                                      float *surround_out, size_t frames) {
  if (_current_method.load(std::memory_order_acquire) == UpmixMethod::Matrix) {
    _impl->matrix_upmix(stereo_in, surround_out, frames);
  } else {
    _impl->hrtf_upmix(stereo_in, surround_out, frames);
  }
}

void MultichannelUpmixEngine::set_upmix_method(UpmixMethod method) noexcept {
  _current_method.store(method, std::memory_order_release);
}

void MultichannelUpmixEngine::reset() noexcept {
  std::fill(_impl->delay_buffer.begin(), _impl->delay_buffer.end(), 0.0f);
  _impl->write_index = 0;
}
