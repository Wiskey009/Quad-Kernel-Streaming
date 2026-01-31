#include "audio_preprocessing_chain.h"

namespace audio_processing {

Normalizer::Normalizer(float target_db)
    : target_amplitude_(powf(10.0f, target_db / 20.0f)) {}

void Normalizer::process(float *data, size_t num_samples) noexcept {
  float peak = find_peak(data, num_samples);
  if (peak > 1e-6f) {
    apply_gain(data, num_samples, target_amplitude_ / peak);
  }
}

float Normalizer::find_peak(const float *data, size_t num_samples) noexcept {
  float max_peak = 0.0f;
  size_t i = 0;
#ifdef APC_USE_AVX
  __m256 v_max = _mm256_set1_ps(0.0f);
  __m256 v_abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  for (; i + 7 < num_samples; i += 8) {
    __m256 v_in = _mm256_loadu_ps(&data[i]);
    __m256 v_abs = _mm256_and_ps(v_in, v_abs_mask);
    v_max = _mm256_max_ps(v_max, v_abs);
  }
  float tmp[8];
  _mm256_storeu_ps(tmp, v_max);
  for (int j = 0; j < 8; ++j)
    max_peak = std::max(max_peak, tmp[j]);
#endif
  for (; i < num_samples; ++i)
    max_peak = std::max(max_peak, fabsf(data[i]));
  return max_peak;
}

void Normalizer::apply_gain(float *data, size_t num_samples,
                            float gain) noexcept {
  size_t i = 0;
#ifdef APC_USE_AVX
  __m256 v_gain = _mm256_set1_ps(gain);
  for (; i + 7 < num_samples; i += 8) {
    __m256 v_in = _mm256_loadu_ps(&data[i]);
    _mm256_storeu_ps(&data[i], _mm256_mul_ps(v_in, v_gain));
  }
#endif
  for (; i < num_samples; ++i)
    data[i] *= gain;
}

Compressor::Compressor(float threshold_db, float ratio, float attack_ms,
                       float release_ms, float sample_rate) {
  threshold_ = powf(10.0f, threshold_db / 20.0f);
  ratio_ = 1.0f / ratio;
  attack_coeff_ = expf(-1.0f / (attack_ms * sample_rate * 0.001f));
  release_coeff_ = expf(-1.0f / (release_ms * sample_rate * 0.001f));
}

void Compressor::process(float *data, size_t num_samples) noexcept {
  for (size_t i = 0; i < num_samples; ++i) {
    float abs_sample = fabsf(data[i]);
    if (abs_sample > envelope_) {
      envelope_ =
          attack_coeff_ * envelope_ + (1.0f - attack_coeff_) * abs_sample;
    } else {
      envelope_ =
          release_coeff_ * envelope_ + (1.0f - release_coeff_) * abs_sample;
    }

    if (envelope_ > threshold_) {
      float gain = threshold_ + (envelope_ - threshold_) * ratio_;
      data[i] *= (gain / envelope_);
    }
  }
}

EQFilter::EQFilter(float center_freq, float gain_db, float q,
                   float sample_rate) {
  float w0 = 2.0f * (float)M_PI * center_freq / sample_rate;
  float alpha = sinf(w0) / (2.0f * q);
  float A = powf(10.0f, gain_db / 40.0f);
  float cosw0 = cosf(w0);

  float b0 = 1.0f + alpha * A;
  float b1 = -2.0f * cosw0;
  float b2 = 1.0f - alpha * A;
  float a0 = 1.0f + alpha / A;
  float a1 = -2.0f * cosw0;
  float a2 = 1.0f - alpha / A;

  coeffs_.a0 = b0 / a0;
  coeffs_.a1 = b1 / a0;
  coeffs_.a2 = b2 / a0;
  coeffs_.b1 = a1 / a0;
  coeffs_.b2 = a2 / a0;
}

void EQFilter::process(float *data, size_t num_samples) noexcept {
  for (size_t i = 0; i < num_samples; ++i) {
    float x = data[i];
    float y = coeffs_.a0 * x + z_[0];
    z_[0] = coeffs_.a1 * x + z_[1] - coeffs_.b1 * y;
    z_[1] = coeffs_.a2 * x - coeffs_.b2 * y;
    data[i] = y;
  }
}

} // namespace audio_processing
