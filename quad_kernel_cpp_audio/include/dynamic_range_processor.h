#pragma once
#include <cmath>
#include <cstdint>
#include <memory>


#if defined(__AVX__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define DRP_USE_AVX
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define DRP_USE_NEON
#endif

class DynamicRangeProcessor {
public:
  enum class Mode { COMPRESSOR, EXPANDER, LIMITER };

  DynamicRangeProcessor(float thresholdDB, float ratio, float attackMs,
                        float releaseMs, float sampleRate, Mode mode);
  ~DynamicRangeProcessor();

  void process(const float *input, float *output, size_t numSamples);
  void updateParameters(float thresholdDB, float ratio, float attackMs,
                        float releaseMs);

private:
  void processCompressor(const float *input, float *output, size_t numSamples);

  void recalculateCoefficients();

  struct State {
    float envelope = 0.0f;
    float gain = 1.0f;
  };

  float threshold_;
  float ratio_;
  float attackMs_;
  float releaseMs_;
  float attackCoeff_;
  float releaseCoeff_;
  float sampleRate_;
  Mode mode_;

  State serialState_;
};
