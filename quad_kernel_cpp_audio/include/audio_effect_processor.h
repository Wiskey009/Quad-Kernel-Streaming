#pragma once
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#if defined(__AVX__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define AEP_USE_AVX
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define AEP_USE_NEON
#endif

namespace dsp {

enum class EffectType { REVERB, DELAY, CHORUS, DISTORTION, PARAMETRIC_EQ };

struct EffectParams {
  std::atomic<float> wet_dry{0.5f};
  std::atomic<float> main_param[4]{0.0f, 0.0f, 0.0f, 0.0f};
};

class AudioEffectProcessor {
public:
  explicit AudioEffectProcessor(size_t max_block_size, double sample_rate);
  ~AudioEffectProcessor();

  void processBlock(float *in_out, size_t num_samples);
  void setEffectType(EffectType type);
  void updateParameters(const EffectParams &params);

private:
  void processReverb(float *in_out, size_t num_samples);
  void processDelay(float *in_out, size_t num_samples);
  void processChorus(float *in_out, size_t num_samples);
  void processDistortion(float *in_out, size_t num_samples);
  void processParametricEQ(float *in_out, size_t num_samples);

  class CircularBuffer;
  class BiquadFilter {
  public:
    void configure(float freq, float Q, float gain_db, double sample_rate);
    float process(float in);

  private:
    float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
    float x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  };

  const double sample_rate_;
  const size_t max_block_size_;
  EffectType current_effect_ = EffectType::REVERB;
  EffectParams params_;

  std::vector<float> temp_buffer_;
  std::unique_ptr<CircularBuffer> delay_buffer_;
  std::array<BiquadFilter, 2> eq_filters_;
};

} // namespace dsp
