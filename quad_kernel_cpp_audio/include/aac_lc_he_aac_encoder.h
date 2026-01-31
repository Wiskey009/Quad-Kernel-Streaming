#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>


#if defined(__AVX__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define AAC_USE_AVX
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define AAC_USE_NEON
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace aac {

struct EncoderConfig {
  int sample_rate;
  int channels;
  int bitrate;
  bool he_aac_enabled;
  bool sbr_ratio_mode;
};

class AACFrame {
public:
  AACFrame() = default;
  AACFrame(const AACFrame &) = delete;
  AACFrame &operator=(const AACFrame &) = delete;

  AACFrame(AACFrame &&) noexcept = default;
  AACFrame &operator=(AACFrame &&) noexcept = default;

  const std::vector<uint8_t> &data() const { return data_; }
  uint64_t pts() const { return pts_; }

  std::vector<uint8_t> data_;
  uint64_t pts_ = 0;
};

class AACEncoder {
public:
  explicit AACEncoder(const EncoderConfig &config);
  ~AACEncoder();

  void encode(const float *pcm, size_t samples,
              std::vector<AACFrame> &output) noexcept;

  void preallocate_resources();

private:
  void analysis_filterbank_avx(const float *audio_in);
  void analysis_filterbank_neon(const float *audio_in);

  void psychoacoustic_model();
  void quantize_spectrum();
  void huffman_coding();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace aac
