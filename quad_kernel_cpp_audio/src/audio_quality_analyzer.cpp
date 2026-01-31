#include "../include/audio_quality_analyzer.h"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

// Note: In a production environment, fftw3.h must be available.
// We'll mock the calls or include the header if possible.
#define _USE_MATH_DEFINES
#include <cmath>

#if __has_include(<fftw3.h>)
#include <fftw3.h>
#else
// Mock for compilation if header missing
// Types are already defined in the header (audio_quality_analyzer.h)
#define FFTW_MEASURE 0
extern "C" {
void *fftw_malloc(size_t s) { return malloc(s); }
void fftw_free(void *p) { free(p); }
void fftw_destroy_plan(fftw_plan) {}
fftw_plan fftw_plan_dft_r2c_1d(int, double *, fftw_complex *, unsigned) {
  return nullptr;
}
void fftw_execute(fftw_plan) {}
}
#endif

namespace audio_quality {

void AudioQualityAnalyzer::FFTWPlanDeleter::operator()(fftw_plan p) const {
  if (p)
    fftw_destroy_plan(p);
}
void AudioQualityAnalyzer::FFTWDataDeleter::operator()(void *p) const {
  if (p)
    fftw_free(p);
}

AudioQualityAnalyzer::AudioQualityAnalyzer(int sample_rate, int frame_size)
    : sample_rate_(sample_rate), frame_size_(frame_size),
      spectrum_bins_(frame_size_ / 2 + 1) {

  fft_in_.reset(
      static_cast<double *>(fftw_malloc(sizeof(double) * frame_size_)));
  fft_out_.reset(static_cast<fftw_complex *>(
      fftw_malloc(sizeof(fftw_complex) * spectrum_bins_)));

  if (!fft_in_ || !fft_out_)
    throw std::runtime_error("FFTW allocation failed");

  fft_plan_.reset(fftw_plan_dft_r2c_1d(frame_size_, fft_in_.get(),
                                       fft_out_.get(), FFTW_MEASURE));

  window_.resize(frame_size_);
  for (int i = 0; i < frame_size_; ++i) {
    window_[i] =
        0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (frame_size_ - 1)));
  }

  spectrum_.resize(spectrum_bins_, 0.0);
}

// Destructor is now empty as smart pointers handle cleanup
AudioQualityAnalyzer::~AudioQualityAnalyzer() = default;

void AudioQualityAnalyzer::process_reference(const float *data, size_t size) {
  if (size != static_cast<size_t>(frame_size_))
    throw std::invalid_argument("Invalid frame size");
  reference_.assign(data, data + size);
  reference_loaded_ = true;
}

void AudioQualityAnalyzer::process_degraded(const float *data, size_t size) {
  if (size != static_cast<size_t>(frame_size_))
    throw std::invalid_argument("Invalid frame size");
  degraded_.assign(data, data + size);
  degraded_loaded_ = true;

  if (reference_loaded_) {
    time_align_signals();
    calculate_pesq();
    calculate_stoi();
    compute_spectral_envelope();
  }
}

void AudioQualityAnalyzer::apply_window(float *data) noexcept {
  size_t i = 0;
#ifdef AQA_USE_AVX
  for (; i + 7 < static_cast<size_t>(frame_size_); i += 8) {
    __m256 d = _mm256_loadu_ps(&data[i]);
    __m256 w = _mm256_loadu_ps(&window_[i]);
    _mm256_storeu_ps(&data[i], _mm256_mul_ps(d, w));
  }
#endif
  for (; i < static_cast<size_t>(frame_size_); ++i) {
    data[i] *= window_[i];
  }
}

void AudioQualityAnalyzer::time_align_signals() noexcept {
  size_t best_lag = 0;
  double max_corr = -1.0e20;
  for (size_t lag = 0; lag < 100 && lag < reference_.size(); ++lag) {
    double corr = 0.0;
    for (size_t i = lag; i < reference_.size(); ++i) {
      corr += reference_[i] * degraded_[i - lag];
    }
    if (corr > max_corr) {
      max_corr = corr;
      best_lag = lag;
    }
  }
  std::rotate(degraded_.begin(), degraded_.begin() + best_lag, degraded_.end());
}

void AudioQualityAnalyzer::compute_spectral_envelope() noexcept {
  for (int i = 0; i < frame_size_; ++i)
    fft_in_.get()[i] = static_cast<double>(reference_[i] * window_[i]);
  fftw_execute(fft_plan_.get());
  for (int i = 0; i < spectrum_bins_; ++i) {
    spectrum_[i] = sqrt(fft_out_.get()[i][0] * fft_out_.get()[i][0] +
                        fft_out_.get()[i][1] * fft_out_.get()[i][1]);
  }
}

void AudioQualityAnalyzer::calculate_pesq() {
  pesq_score_ = 4.0; // Placeholder for actual implementation
}

void AudioQualityAnalyzer::calculate_stoi() {
  stoi_score_ = 0.95; // Placeholder for actual implementation
}

double AudioQualityAnalyzer::get_pesq() const { return pesq_score_; }
double AudioQualityAnalyzer::get_stoi() const { return stoi_score_; }
const std::vector<double> &AudioQualityAnalyzer::get_spectrum() const {
  return spectrum_;
}

} // namespace audio_quality
