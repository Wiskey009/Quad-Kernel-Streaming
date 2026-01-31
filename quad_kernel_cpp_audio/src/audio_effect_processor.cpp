#include "audio_effect_processor.h"
#include <algorithm>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace dsp {

class AudioEffectProcessor::CircularBuffer {
public:
  CircularBuffer(size_t size) : buffer_(size), write_ptr_(0) {}

  void write(const float *data, size_t samples) {
    const size_t first_part = std::min(samples, buffer_.size() - write_ptr_);
    std::memcpy(buffer_.data() + write_ptr_, data, first_part * sizeof(float));
    if (samples > first_part) {
      std::memcpy(buffer_.data(), data + first_part,
                  (samples - first_part) * sizeof(float));
    }
    write_ptr_ = (write_ptr_ + samples) % buffer_.size();
  }

  void read(float *out, size_t delay_samples, size_t num_samples) {
    size_t read_ptr =
        (write_ptr_ - delay_samples + buffer_.size()) % buffer_.size();
    for (size_t i = 0; i < num_samples; ++i) {
      out[i] = buffer_[read_ptr];
      read_ptr = (read_ptr + 1) % buffer_.size();
    }
  }

private:
  std::vector<float> buffer_;
  size_t write_ptr_;
};

void AudioEffectProcessor::BiquadFilter::configure(float freq, float Q,
                                                   float gain_db,
                                                   double sample_rate) {
  const float A = powf(10.0f, gain_db / 40.0f);
  const float omega = static_cast<float>(2.0 * M_PI * freq / sample_rate);
  const float sn = sinf(omega);
  const float cs = cosf(omega);
  const float alpha = sn / (2.0f * Q);

  float b0_raw = 1.0f + (alpha * A);
  float b1_raw = -2.0f * cs;
  float b2_raw = 1.0f - (alpha * A);
  float a0 = 1.0f + (alpha / A);
  float a1_raw = -2.0f * cs;
  float a2_raw = 1.0f - (alpha / A);

  const float norm = 1.0f / a0;
  this->b0 = b0_raw * norm;
  this->b1 = b1_raw * norm;
  this->b2 = b2_raw * norm;
  this->a1 = a1_raw * norm;
  this->a2 = a2_raw * norm;
}

float AudioEffectProcessor::BiquadFilter::process(float in) {
  const float out = in * b0 + x1 * b1 + x2 * b2 - y1 * a1 - y2 * a2;
  x2 = x1;
  x1 = in;
  y2 = y1;
  y1 = out;
  return out;
}

AudioEffectProcessor::AudioEffectProcessor(size_t max_block_size,
                                           double sample_rate)
    : sample_rate_(sample_rate), max_block_size_(max_block_size) {
  temp_buffer_.resize(max_block_size, 0.0f);
  delay_buffer_ = std::make_unique<CircularBuffer>(
      static_cast<size_t>(sample_rate * 2.0)); // 2 sec max delay
  for (auto &f : eq_filters_)
    f.configure(1000.0f, 0.707f, 0.0f, sample_rate);
}

AudioEffectProcessor::~AudioEffectProcessor() = default;

void AudioEffectProcessor::processBlock(float *in_out, size_t num_samples) {
  if (num_samples > max_block_size_)
    return;
  switch (current_effect_) {
  case EffectType::REVERB:
    processReverb(in_out, num_samples);
    break;
  case EffectType::DELAY:
    processDelay(in_out, num_samples);
    break;
  case EffectType::CHORUS:
    processChorus(in_out, num_samples);
    break;
  case EffectType::DISTORTION:
    processDistortion(in_out, num_samples);
    break;
  case EffectType::PARAMETRIC_EQ:
    processParametricEQ(in_out, num_samples);
    break;
  }
}

void AudioEffectProcessor::processReverb(float *in_out, size_t num_samples) {
  const float wet = params_.wet_dry.load(std::memory_order_relaxed);
  const float dry = 1.0f - wet;
  const float decay = params_.main_param[0].load(std::memory_order_relaxed);

  for (size_t i = 0; i < num_samples; ++i) {
    float input = in_out[i];
    float delayed = temp_buffer_[i];
    float processed = (input + delayed) * decay * 0.7f;
    temp_buffer_[i] = processed;
    in_out[i] = (input * dry) + (processed * wet);
  }
}

void AudioEffectProcessor::processDelay(float *in_out, size_t num_samples) {
  const float feedback = params_.main_param[0].load(std::memory_order_relaxed);
  const size_t delay_s = static_cast<size_t>(
      params_.main_param[1].load(std::memory_order_relaxed) * sample_rate_);

  delay_buffer_->read(temp_buffer_.data(), delay_s, num_samples);
  for (size_t i = 0; i < num_samples; ++i) {
    float out = in_out[i] + temp_buffer_[i] * 0.5f;
    float next_delay = (in_out[i] + temp_buffer_[i] * feedback);
    in_out[i] = out;
    temp_buffer_[i] = next_delay;
  }
  delay_buffer_->write(temp_buffer_.data(), num_samples);
}

void AudioEffectProcessor::processChorus(float *in_out, size_t num_samples) {
  (void)in_out;
  (void)num_samples;
}
void AudioEffectProcessor::processDistortion(float *in_out,
                                             size_t num_samples) {
  const float drive =
      params_.main_param[0].load(std::memory_order_relaxed) * 10.0f;
  for (size_t i = 0; i < num_samples; ++i) {
    in_out[i] = tanhf(in_out[i] * drive);
  }
}
void AudioEffectProcessor::processParametricEQ(float *in_out,
                                               size_t num_samples) {
  for (size_t i = 0; i < num_samples; ++i) {
    in_out[i] = eq_filters_[0].process(in_out[i]);
  }
}

void AudioEffectProcessor::setEffectType(EffectType type) {
  current_effect_ = type;
}
void AudioEffectProcessor::updateParameters(const EffectParams &p) {
  params_.wet_dry.store(p.wet_dry.load());
  for (int i = 0; i < 4; ++i)
    params_.main_param[i].store(p.main_param[i].load());
}

} // namespace dsp
