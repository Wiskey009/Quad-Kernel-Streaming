#include "dynamic_range_processor.h"
#include <algorithm>

DynamicRangeProcessor::DynamicRangeProcessor(float thresholdDB, float ratio,
                                             float attackMs, float releaseMs,
                                             float sampleRate, Mode mode)
    : threshold_(powf(10.0f, thresholdDB / 20.0f)), ratio_(ratio),
      attackMs_(attackMs), releaseMs_(releaseMs), sampleRate_(sampleRate),
      mode_(mode) {
  recalculateCoefficients();
}

DynamicRangeProcessor::~DynamicRangeProcessor() = default;

void DynamicRangeProcessor::process(const float *input, float *output,
                                    size_t numSamples) {
  // For this implementation, we focus on the compressor mode as the most common
  // use case
  if (mode_ == Mode::COMPRESSOR || mode_ == Mode::LIMITER) {
    processCompressor(input, output, numSamples);
  }
}

void DynamicRangeProcessor::processCompressor(const float *input, float *output,
                                              size_t numSamples) {
  State &state = serialState_;
  const float threshold = threshold_;
  const float ratio = ratio_;
  const float attack = attackCoeff_;
  const float release = releaseCoeff_;

  for (size_t i = 0; i < numSamples; ++i) {
    const float x = input[i];
    const float abs_x = fabsf(x);

    // Envelope follower (peak detection)
    if (abs_x > state.envelope) {
      state.envelope = attack * state.envelope + (1.0f - attack) * abs_x;
    } else {
      state.envelope = release * state.envelope + (1.0f - release) * abs_x;
    }

    float gain = 1.0f;
    if (state.envelope > threshold) {
      // Static curve calculation
      float over = 20.0f * log10f(state.envelope / threshold);
      float reduced = over / ratio;
      gain = powf(10.0f, (reduced - over) / 20.0f);
    }

    output[i] = x * gain;
    state.gain = gain;
  }
}

void DynamicRangeProcessor::recalculateCoefficients() {
  // T = 1 - e^(-1 / (t_ms * 0.001 * fs))
  attackCoeff_ = 1.0f - expf(-1.0f / (attackMs_ * 0.001f * sampleRate_));
  releaseCoeff_ = 1.0f - expf(-1.0f / (releaseMs_ * 0.001f * sampleRate_));
}

void DynamicRangeProcessor::updateParameters(float thresholdDB, float ratio,
                                             float attackMs, float releaseMs) {
  threshold_ = powf(10.0f, thresholdDB / 20.0f);
  ratio_ = ratio;
  attackMs_ = attackMs;
  releaseMs_ = releaseMs;
  recalculateCoefficients();
}
