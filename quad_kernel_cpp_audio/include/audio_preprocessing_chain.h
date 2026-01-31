#pragma once
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>


#if defined(__AVX__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define APC_USE_AVX
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define APC_USE_NEON
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace audio_processing {

class AudioProcessor {
public:
  virtual ~AudioProcessor() = default;
  virtual void process(float *data, size_t num_samples) noexcept = 0;
  virtual void reset() noexcept = 0;
};

class Normalizer : public AudioProcessor {
public:
  explicit Normalizer(float target_db = -1.0f);
  void process(float *data, size_t num_samples) noexcept override;
  void reset() noexcept override {}

private:
  float target_amplitude_;
  float find_peak(const float *data, size_t num_samples) noexcept;
  void apply_gain(float *data, size_t num_samples, float gain) noexcept;
};

class Compressor : public AudioProcessor {
public:
  Compressor(float threshold_db, float ratio, float attack_ms, float release_ms,
             float sample_rate);
  void process(float *data, size_t num_samples) noexcept override;
  void reset() noexcept override { envelope_ = 0.0f; }

private:
  float threshold_;
  float ratio_;
  float attack_coeff_;
  float release_coeff_;
  float envelope_ = 0.0f;
};

class EQFilter : public AudioProcessor {
public:
  EQFilter(float center_freq, float gain_db, float q, float sample_rate);
  void process(float *data, size_t num_samples) noexcept override;
  void reset() noexcept override { z_[0] = z_[1] = 0.0f; }

private:
  struct {
    float a0, a1, a2, b1, b2;
  } coeffs_;
  float z_[2] = {0.0f};
};

class ProcessingChain {
public:
  void add_processor(std::unique_ptr<AudioProcessor> processor) {
    processors_.emplace_back(std::move(processor));
  }
  void process(float *data, size_t num_samples) noexcept {
    for (auto &proc : processors_)
      proc->process(data, num_samples);
  }
  void reset() noexcept {
    for (auto &proc : processors_)
      proc->reset();
  }

private:
  std::vector<std::unique_ptr<AudioProcessor>> processors_;
};

} // namespace audio_processing
