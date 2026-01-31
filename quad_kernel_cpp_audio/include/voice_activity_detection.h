#pragma once
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>


#if defined(__AVX__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define VAD_USE_AVX
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define VAD_USE_NEON
#endif

struct VADConfig {
  float sample_rate;
  int frame_size_ms;
  float aggressiveness;
  float noise_floor_decay;
  int hangover_frames;
};

class VoiceActivityDetector {
public:
  explicit VoiceActivityDetector(const VADConfig &config);
  ~VoiceActivityDetector();

  bool process(const float *audio);
  void update_config(const VADConfig &new_config) noexcept;

private:
  struct FFTState;
  void extract_features(const float *frame) noexcept;
  float calculate_spectral_flatness() const noexcept;
  float calculate_normalized_energy() const noexcept;
  void update_state_machine(float feature) noexcept;

  std::unique_ptr<FFTState> fft_engine_;
  std::vector<float> fft_magnitudes_;
  std::vector<float> audio_buffer_;
  size_t buffer_pos_ = 0;

  std::atomic<float> energy_threshold_;
  std::atomic<float> spectral_threshold_;
  std::atomic<int> hangover_count_;

  float noise_floor_ = 1e-9f;
  int speech_counter_ = 0;
  bool speech_detected_ = false;
  static constexpr int kMaxFrameSize = 1536;
};
