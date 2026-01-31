#pragma once
#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>


enum class UpmixMethod { Matrix, HRTF };

class MultichannelUpmixEngine {
public:
  explicit MultichannelUpmixEngine(size_t output_channels, float sample_rate,
                                   UpmixMethod method);
  ~MultichannelUpmixEngine();

  void process(const float *stereo_in, float *surround_out, size_t frames);
  void set_upmix_method(UpmixMethod method) noexcept;
  void reset() noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
  std::atomic<UpmixMethod> _current_method;
};
