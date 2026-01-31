#include "voice_activity_detection.h"
#include <algorithm>
#include <numeric>

struct VoiceActivityDetector::FFTState {
  void RealFFT(float *data, float *spectrum_out, size_t size) noexcept {
    // Simplified magnitude spectrum estimation
    for (size_t i = 0; i < size / 2 + 1; ++i) {
      spectrum_out[i] = fabsf(data[i]);
    }
  }
};

VoiceActivityDetector::VoiceActivityDetector(const VADConfig &config)
    : fft_magnitudes_(kMaxFrameSize / 2 + 1), audio_buffer_(kMaxFrameSize) {
  fft_engine_ = std::make_unique<FFTState>();
  update_config(config);
}

VoiceActivityDetector::~VoiceActivityDetector() = default;

void VoiceActivityDetector::update_config(
    const VADConfig &new_config) noexcept {
  const float base_thresh = expf(-new_config.aggressiveness);
  energy_threshold_.store(base_thresh, std::memory_order_relaxed);
  spectral_threshold_.store(0.5f - (0.1f * new_config.aggressiveness),
                            std::memory_order_relaxed);
  hangover_count_.store(new_config.hangover_frames, std::memory_order_relaxed);
}

bool VoiceActivityDetector::process(const float *audio) {
  const size_t frames = 480; // 10ms at 48kHz
  std::copy_n(audio, frames, audio_buffer_.data());
  extract_features(audio_buffer_.data());
  return speech_detected_;
}

void VoiceActivityDetector::extract_features(const float *frame) noexcept {
  fft_engine_->RealFFT(const_cast<float *>(frame), fft_magnitudes_.data(),
                       kMaxFrameSize);

  const float energy = calculate_normalized_energy();
  const float flatness = calculate_spectral_flatness();
  const float combined = 0.7f * energy + 0.3f * (1.0f - flatness);

  noise_floor_ = std::max(1e-9f, noise_floor_ * 0.999f + combined * 0.001f);
  const float normalized = combined / noise_floor_;
  update_state_machine(normalized);
}

float VoiceActivityDetector::calculate_normalized_energy() const noexcept {
  float energy = 0.0f;
  size_t i = 0;
#ifdef VAD_USE_AVX
  __m256 v_sum = _mm256_setzero_ps();
  for (; i + 7 < fft_magnitudes_.size(); i += 8) {
    __m256 v_mag = _mm256_loadu_ps(&fft_magnitudes_[i]);
    v_sum = _mm256_add_ps(v_sum, _mm256_mul_ps(v_mag, v_mag));
  }
  float tmp[8];
  _mm256_storeu_ps(tmp, v_sum);
  for (float t : tmp)
    energy += t;
#endif
  for (; i < fft_magnitudes_.size(); ++i) {
    energy += fft_magnitudes_[i] * fft_magnitudes_[i];
  }
  return log1pf(energy);
}

float VoiceActivityDetector::calculate_spectral_flatness() const noexcept {
  // Simplified spectral flatness: geometric mean / arithmetic mean
  double sum = 0.0, log_sum = 0.0;
  const size_t n = fft_magnitudes_.size();
  for (float mag : fft_magnitudes_) {
    float val = std::max(mag, 1e-9f);
    sum += val;
    log_sum += logf(val);
  }
  float arithmetic = static_cast<float>(sum / n);
  float geometric = expf(static_cast<float>(log_sum / n));
  return geometric / (arithmetic + 1e-9f);
}

void VoiceActivityDetector::update_state_machine(float feature) noexcept {
  const float energy_thresh = energy_threshold_.load(std::memory_order_relaxed);
  if (feature > energy_thresh) {
    speech_counter_ = std::min(speech_counter_ + 5, hangover_count_.load());
  } else {
    speech_counter_ = std::max(speech_counter_ - 1, 0);
  }
  speech_detected_ = (speech_counter_ > 0);
}
