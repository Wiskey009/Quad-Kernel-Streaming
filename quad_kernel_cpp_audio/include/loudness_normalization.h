#pragma once
#include <cstddef>
#include <memory>
#include <vector>


class LoudnessNormalizer {
public:
  explicit LoudnessNormalizer(size_t sample_rate = 48000);
  ~LoudnessNormalizer();

  void process(const float *input, size_t num_samples);
  double get_loudness_lkfs() const;
  void reset();

private:
  struct FilterState;
  std::unique_ptr<FilterState> state_;
  static constexpr double GATE_THRESHOLD = -70.0;
};
