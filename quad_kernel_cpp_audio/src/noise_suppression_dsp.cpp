#include "noise_suppression_dsp.h"
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace dsp {

// Mock FFT implementation for compilation safety
struct RealTimeFFT {
  int fft_size;
  explicit RealTimeFFT(int size) : fft_size(size) {}
  void forward(float *, float *) {}
  void inverse(float *, float *) {}
};

struct NoiseSuppressor::Impl {
  Config config;
  int fft_size;
  int overlap;
  std::vector<float> window;
  std::vector<float> overlap_buffer;
  std::unique_ptr<RealTimeFFT> fft_processor;
  NoiseEstimator ml_estimator = nullptr;
  void *ml_context = nullptr;

  alignas(32) std::vector<float> fft_real;
  alignas(32) std::vector<float> fft_imag;
  alignas(32) std::vector<float> spectrum_mag;

  explicit Impl(const Config &cfg) : config(cfg) {
    fft_size = cfg.frame_size;
    overlap = cfg.frame_size / cfg.overlap_factor;

    window.resize(fft_size);
    const float scale = static_cast<float>(2.0 * M_PI / fft_size);
    for (int i = 0; i < fft_size; ++i) {
      window[i] = 0.5f * (1.0f - cosf(scale * i));
    }

    fft_processor = std::make_unique<RealTimeFFT>(fft_size);
    fft_real.resize(fft_size);
    fft_imag.resize(fft_size);
    spectrum_mag.resize(fft_size / 2 + 1);
    overlap_buffer.resize(fft_size, 0.0f);
  }

  void apply_spectral_subtraction() {
    const int num_bins = fft_size / 2 + 1;
    float noise_floor = powf(10.0f, config.noise_gate_db / 20.0f);

    if (ml_estimator) {
      noise_floor = ml_estimator(spectrum_mag.data(), num_bins, ml_context);
    }

#if defined(DSP_USE_AVX)
    const __m256 noise_vec = _mm256_set1_ps(noise_floor);
    const __m256 min_mag = _mm256_set1_ps(1e-10f);
    for (int i = 0; i + 7 < num_bins; i += 8) {
      __m256 mag = _mm256_loadu_ps(&spectrum_mag[i]);
      __m256 masked = _mm256_sub_ps(mag, noise_vec);
      masked = _mm256_max_ps(masked, min_mag);
      _mm256_storeu_ps(&spectrum_mag[i], masked);
    }
#else
    for (int i = 0; i < num_bins; ++i) {
      spectrum_mag[i] = std::max(spectrum_mag[i] - noise_floor, 1e-10f);
    }
#endif
  }
};

NoiseSuppressor::NoiseSuppressor(const Config &cfg)
    : impl_(std::make_unique<Impl>(cfg)) {}
NoiseSuppressor::~NoiseSuppressor() = default;

void NoiseSuppressor::process(float *input, float *output, int num_channels,
                              int num_frames) {
  if (num_channels != 1)
    return;

  auto &im = *impl_;
  for (int i = 0; i < num_frames; i += im.fft_size - im.overlap) {
    // Windowing
    for (int k = 0; k < im.fft_size; ++k) {
      im.fft_real[k] = input[i + k] * im.window[k];
    }

    im.fft_processor->forward(im.fft_real.data(), im.fft_imag.data());

    // Magnitude
    const int num_bins = im.fft_size / 2 + 1;
    for (int k = 0; k < num_bins; ++k) {
      float re = im.fft_real[k];
      float img = im.fft_imag[k];
      im.spectrum_mag[k] = sqrtf(re * re + img * img);
    }

    im.apply_spectral_subtraction();

    // Reconstruct and Overlap-Add
    im.fft_processor->inverse(im.fft_real.data(), im.fft_imag.data());

    for (int k = 0; k < im.fft_size; ++k) {
      output[i + k] = im.fft_real[k] + im.overlap_buffer[k];
      im.overlap_buffer[k] =
          (k < im.fft_size - im.overlap) ? im.fft_real[k + im.overlap] : 0.0f;
    }
  }
}

void NoiseSuppressor::set_noise_estimator(NoiseEstimator estimator,
                                          void *context) {
  impl_->ml_estimator = estimator;
  impl_->ml_context = context;
}

NoiseSuppressor::NoiseSuppressor(NoiseSuppressor &&) noexcept = default;
NoiseSuppressor &
NoiseSuppressor::operator=(NoiseSuppressor &&) noexcept = default;

} // namespace dsp
