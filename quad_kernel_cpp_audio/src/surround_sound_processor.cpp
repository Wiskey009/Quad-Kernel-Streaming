#include "surround_sound_processor.h"
#include <algorithm>
#include <cassert>
#include <cstring>

namespace dsp {

SurroundSoundProcessor::SurroundSoundProcessor(ChannelLayout output_layout)
    : layout_(output_layout), coefficients_dirty_(false) {

  // Initialize coefficient buffer
  aligned_coefficients_.resize(64);

  const size_t channels = (output_layout == ChannelLayout::LAYOUT_51) ? 6 : 8;

  // Reserve vectors
  input_buffers_.resize(channels);
  output_buffers_.resize(channels);

  for (size_t i = 0; i < channels; ++i) {
    input_buffers_[i].resize(4096);
    output_buffers_[i].resize(4096);
  }

  mixing_coefficients_.resize(channels * channels, 0.0f);
  for (size_t i = 0; i < channels; ++i)
    mixing_coefficients_[i * channels + i] = 1.0f;
}

SurroundSoundProcessor::~SurroundSoundProcessor() = default;

SurroundSoundProcessor::SurroundSoundProcessor(
    SurroundSoundProcessor &&) noexcept = default;
SurroundSoundProcessor &
SurroundSoundProcessor::operator=(SurroundSoundProcessor &&) noexcept = default;

void SurroundSoundProcessor::process(const float **input_buffers,
                                     float **output_buffers,
                                     size_t num_frames) {
  const size_t num_channels = input_buffers_.size();
  for (size_t i = 0; i < num_channels; ++i) {
    if (num_frames <= input_buffers_[i].size()) {
      std::memcpy(input_buffers_[i].data(), input_buffers[i],
                  num_frames * sizeof(float));
    }
  }
  encode_51(input_buffers, output_buffers, num_frames);
  apply_mixing(output_buffers, num_frames);
}

void SurroundSoundProcessor::encode_51(const float **inputs, float **outputs,
                                       size_t num_frames) {
  const size_t channels = 6;
  const size_t avail_channels = std::min(channels, output_buffers_.size());

  for (size_t ch = 0; ch < avail_channels; ++ch) {
    std::copy_n(input_buffers_[ch].data(), num_frames,
                output_buffers_[ch].data());
  }
}

void SurroundSoundProcessor::apply_mixing(float **outputs, size_t num_frames) {
  const size_t channels = input_buffers_.size();

  // Non-atomic check for demo
  if (coefficients_dirty_) {
    coefficients_dirty_ = false;
    if (aligned_coefficients_.size() >= mixing_coefficients_.size()) {
      std::memcpy(aligned_coefficients_.data(), mixing_coefficients_.data(),
                  mixing_coefficients_.size() * sizeof(float));
    }
  }

  for (size_t out_ch = 0; out_ch < channels; ++out_ch) {
    std::memset(outputs[out_ch], 0, num_frames * sizeof(float));

    for (size_t in_ch = 0; in_ch < channels; ++in_ch) {
      const float *input = input_buffers_[in_ch].data();
      const float coeff = aligned_coefficients_[in_ch * channels + out_ch];

      size_t i = 0;
#ifdef SSP_USE_AVX
      __m256 v_coeff = _mm256_set1_ps(coeff);
      for (; i + 7 < num_frames; i += 8) {
        __m256 v_in = _mm256_loadu_ps(&input[i]);
        __m256 v_out = _mm256_loadu_ps(&outputs[out_ch][i]);
        _mm256_storeu_ps(&outputs[out_ch][i],
                         _mm256_add_ps(v_out, _mm256_mul_ps(v_in, v_coeff)));
      }
#endif
      for (; i < num_frames; ++i)
        outputs[out_ch][i] += input[i] * coeff;
    }
  }
}

void SurroundSoundProcessor::update_mixing_coefficients(
    const std::vector<float> &new_coeffs) {
  assert(new_coeffs.size() == mixing_coefficients_.size());
  std::copy(new_coeffs.begin(), new_coeffs.end(), mixing_coefficients_.begin());
  coefficients_dirty_ = true;
}

} // namespace dsp
