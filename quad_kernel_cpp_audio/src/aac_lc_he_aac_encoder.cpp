#include "aac_lc_he_aac_encoder.h"
#include <algorithm>
#include <array>
#include <stdexcept>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace aac {

struct AACEncoder::Impl {
  EncoderConfig cfg;
  std::vector<float> mdct_buf;
  std::vector<float> sbr_buf;
  std::array<AACFrame, 16> frame_pool;
  alignas(32) std::array<float, 1024> simd_window;

  void reset_buffers() {
    std::fill(mdct_buf.begin(), mdct_buf.end(), 0.0f);
    if (cfg.he_aac_enabled) {
      std::fill(sbr_buf.begin(), sbr_buf.end(), 0.0f);
    }
  }
};

AACEncoder::AACEncoder(const EncoderConfig &config)
    : impl_(std::make_unique<Impl>()) {
  if (config.channels > 8 || config.bitrate < 8000) {
    throw std::invalid_argument("Invalid encoder configuration");
  }

  impl_->cfg = config;
  impl_->mdct_buf.resize(2048 * config.channels);

  if (config.he_aac_enabled) {
    impl_->sbr_buf.resize(4096 * config.channels);
  }

  constexpr int window_size = 1024;
  for (int i = 0; i < window_size; ++i) {
    impl_->simd_window[i] =
        static_cast<float>(std::sin(M_PI * (i + 0.5) / window_size));
  }
}

AACEncoder::~AACEncoder() = default;

void AACEncoder::preallocate_resources() { impl_->reset_buffers(); }

void AACEncoder::encode(const float *pcm, size_t samples,
                        std::vector<AACFrame> &output) noexcept {
  output.clear();

#if defined(AAC_USE_AVX)
  analysis_filterbank_avx(pcm);
#elif defined(AAC_USE_NEON)
  analysis_filterbank_neon(pcm);
#endif

  psychoacoustic_model();
  quantize_spectrum();
  huffman_coding();

  // Mock frame for demonstration
  auto &frame = impl_->frame_pool[0];
  frame.data_.assign(512, 0); // Placeholder
  output.push_back(std::move(frame));
}

void AACEncoder::analysis_filterbank_avx(const float *audio_in) {
#ifdef AAC_USE_AVX
  constexpr int stride = 8;
  auto *window = impl_->simd_window.data();
  auto *buf = impl_->mdct_buf.data();

  for (int i = 0; i < 1024; i += stride) {
    __m256 data = _mm256_loadu_ps(&audio_in[i]);
    __m256 win = _mm256_load_ps(&window[i]);
    __m256 result = _mm256_mul_ps(data, win);

    __m256 hist = _mm256_load_ps(&buf[i]);
    _mm256_store_ps(&buf[i], _mm256_add_ps(hist, result));
  }
#endif
}

void AACEncoder::analysis_filterbank_neon(const float *audio_in) {
#ifdef AAC_USE_NEON
  constexpr int stride = 4;
  auto *window = impl_->simd_window.data();
  auto *buf = impl_->mdct_buf.data();

  for (int i = 0; i < 1024; i += stride) {
    float32x4_t data = vld1q_f32(&audio_in[i]);
    float32x4_t win = vld1q_f32(&window[i]);
    float32x4_t result = vmulq_f32(data, win);

    float32x4_t hist = vld1q_f32(&buf[i]);
    vst1q_f32(&buf[i], vaddq_f32(hist, result));
  }
#endif
}

void AACEncoder::psychoacoustic_model() {}
void AACEncoder::quantize_spectrum() {}
void AACEncoder::huffman_coding() {}

} // namespace aac
