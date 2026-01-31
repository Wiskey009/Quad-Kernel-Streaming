#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>


#if defined(__AVX__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define PSE_USE_AVX
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define PSE_USE_NEON
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace PodcastDSP {

struct BandPass {
  float b0_ = 1, b1_ = 0, b2_ = 0, a1_ = 0, a2_ = 0;
  float x1_ = 0, x2_ = 0, y1_ = 0, y2_ = 0;
  float process(float x);
  void reset();
  void configure(float fc, float q, float sample_rate);
};

class PresenceBoost {
public:
  PresenceBoost(float sample_rate);
  void process(float *data, size_t count);
  void set_intensity(float db_boost);

private:
  float sample_rate_;
  float intensity_;
  float a0_, a1_, b1_;
  alignas(32) float z1_[8] = {0};
  void update_coefficients();
};

class DeEsser {
public:
  DeEsser(float sample_rate);
  void process(float *data, size_t count);
  void set_threshold(float threshold_db);

private:
  float sample_rate_;
  float threshold_;
  BandPass detector_;
  float envelope_ = 0.0f;
  void update_detection();
};

class SpeechClarity {
public:
  explicit SpeechClarity(float sample_rate);
  void process(float *data, size_t count);

private:
  struct CompressorBand {
    float gain_ = 1.0f;
    float process(float x, float detection);
  };
  float sample_rate_;
  std::array<CompressorBand, 3> bands_;
  std::array<BandPass, 3> filters_;
  void init_bands();
};

class SpeechEnhancer {
public:
  SpeechEnhancer(float sample_rate, size_t max_block_size);
  void process_block(float *audio_data, size_t num_samples);
  void set_parameters(float presence_db, float deess_thresh,
                      float clarity_sens);

private:
  PresenceBoost presence_boost_;
  DeEsser deesser_;
  SpeechClarity clarity_;
  static constexpr size_t SIMD_WIDTH = 8;
};

} // namespace PodcastDSP
