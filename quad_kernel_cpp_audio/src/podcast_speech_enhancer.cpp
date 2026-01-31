#include "podcast_speech_enhancer.h"
#include <algorithm>

namespace PodcastDSP {

float BandPass::process(float x) {
  float y = b0_ * x + b1_ * x1_ + b2_ * x2_ - a1_ * y1_ - a2_ * y2_;
  x2_ = x1_;
  x1_ = x;
  y2_ = y1_;
  y1_ = y;
  return y;
}

void BandPass::configure(float fc, float q, float sample_rate) {
  float w0 = 2.0f * (float)M_PI * fc / sample_rate;
  float alpha = sinf(w0) / (2.0f * q);
  float cosw0 = cosf(w0);
  float a0 = 1.0f + alpha;
  b0_ = alpha / a0;
  b1_ = 0 / a0;
  b2_ = -alpha / a0;
  a1_ = -2.0f * cosw0 / a0;
  a2_ = (1.0f - alpha) / a0;
}

PresenceBoost::PresenceBoost(float sample_rate)
    : sample_rate_(sample_rate), intensity_(3.0f) {
  update_coefficients();
}

void PresenceBoost::update_coefficients() {
  const float fc = 4000.0f;
  const float w0 = 2.0f * (float)M_PI * fc / sample_rate_;
  const float A = powf(10.0f, intensity_ / 40.0f);
  const float alpha =
      sinf(w0) / 2.0f * sqrtf((A + 1 / A) * (1 / 0.87f - 1) + 2);
  const float cosw0 = cosf(w0);
  const float sqrtA = sqrtf(A);

  float a0 = (A + 1) + (A - 1) * cosw0 + 2 * sqrtA * alpha;
  b1_ = -2 * cosw0 / a0;
  a0_ = A * ((A + 1) + (A - 1) * cosw0 + 2 * sqrtA * alpha) / a0;
  a1_ = -2 * A * ((A - 1) + (A + 1) * cosw0) / a0;
  // Simplified biquad approx for presence
}

void PresenceBoost::process(float *data, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    float x = data[i];
    float y = x * a0_ + z1_[0];
    z1_[0] = x * a1_ - b1_ * y;
    data[i] = y;
  }
}

DeEsser::DeEsser(float sample_rate)
    : sample_rate_(sample_rate), threshold_(-18.0f) {
  update_detection();
}

void DeEsser::update_detection() {
  detector_.configure(6000.0f, 0.707f, sample_rate_);
}

void DeEsser::process(float *data, size_t count) {
  const float attack = 0.001f;
  const float release = 0.050f;
  const float env_att = expf(-1.0f / (sample_rate_ * attack));
  const float env_rel = expf(-1.0f / (sample_rate_ * release));

  for (size_t i = 0; i < count; ++i) {
    float detect = fabsf(detector_.process(data[i]));
    envelope_ = detect > envelope_
                    ? env_att * envelope_ + (1 - env_att) * detect
                    : env_rel * envelope_ + (1 - env_rel) * detect;

    float db = 20.0f * log10f(envelope_ + 1e-6f);
    float gain =
        db > threshold_ ? powf(10.0f, (threshold_ - db) / 20.0f) : 1.0f;
    data[i] *= gain;
  }
}

SpeechClarity::SpeechClarity(float sample_rate) : sample_rate_(sample_rate) {
  init_bands();
}

void SpeechClarity::init_bands() {
  const float freqs[3] = {500.0f, 2500.0f, 5000.0f};
  for (size_t i = 0; i < 3; ++i)
    filters_[i].configure(freqs[i], 0.5f, sample_rate_);
}

float SpeechClarity::CompressorBand::process(float x, float detection) {
  float target = detection > 0.01f ? 1.0f / sqrtf(detection * 10.0f) : 1.0f;
  target = std::clamp(target, 0.5f, 2.0f);
  gain_ = 0.99f * gain_ + 0.01f * target;
  return x * gain_;
}

void SpeechClarity::process(float *data, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    float x = data[i];
    float sum = 0.0f;
    for (int b = 0; b < 3; ++b) {
      float band = filters_[b].process(x);
      sum += bands_[b].process(band, fabsf(band));
    }
    data[i] = sum * 0.8f + x * 0.2f;
  }
}

SpeechEnhancer::SpeechEnhancer(float sample_rate, size_t /*max_block_size*/)
    : presence_boost_(sample_rate), deesser_(sample_rate),
      clarity_(sample_rate) {}

void SpeechEnhancer::process_block(float *audio_data, size_t num_samples) {
  presence_boost_.process(audio_data, num_samples);
  deesser_.process(audio_data, num_samples);
  clarity_.process(audio_data, num_samples);
}

} // namespace PodcastDSP
