#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#if defined(__AVX__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define SSP_USE_AVX
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define SSP_USE_NEON
#endif

namespace dsp {

enum class ChannelLayout { LAYOUT_51, LAYOUT_71 };

class SurroundSoundProcessor {
public:
  explicit SurroundSoundProcessor(ChannelLayout output_layout);
  ~SurroundSoundProcessor();

  SurroundSoundProcessor(const SurroundSoundProcessor &) = delete;
  SurroundSoundProcessor &operator=(const SurroundSoundProcessor &) = delete;
  SurroundSoundProcessor(SurroundSoundProcessor &&) noexcept;
  SurroundSoundProcessor &operator=(SurroundSoundProcessor &&) noexcept;

  void process(const float **input_buffers, float **output_buffers,
               size_t num_frames);
  void update_mixing_coefficients(const std::vector<float> &new_coeffs);

private:
#ifdef SSP_USE_AVX
  using simd_float = __m256;
  static constexpr size_t SIMD_WIDTH = 8;
#else
  using simd_float = float; // Placeholder
  static constexpr size_t SIMD_WIDTH = 1;
#endif

  ChannelLayout layout_;
  bool coefficients_dirty_;
  std::vector<float> mixing_coefficients_;
  std::vector<float> aligned_coefficients_;

  std::vector<std::vector<float>> input_buffers_;
  std::vector<std::vector<float>> output_buffers_;

  void encode_51(const float **inputs, float **outputs, size_t num_frames);
  void apply_mixing(float **outputs, size_t num_frames);
};

} // namespace dsp
