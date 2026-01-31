#pragma once
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#if defined(__AVX__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define AQA_USE_AVX
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define AQA_USE_NEON
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Forward declarations for FFTW types if not included
struct fftw_plan_s;
typedef struct fftw_plan_s *fftw_plan;
typedef double fftw_complex[2];

namespace audio_quality {

class AudioQualityAnalyzer {
public:
  AudioQualityAnalyzer(int sample_rate, int frame_size);
  ~AudioQualityAnalyzer();
  AudioQualityAnalyzer(const AudioQualityAnalyzer &) = delete;
  AudioQualityAnalyzer &operator=(const AudioQualityAnalyzer &) = delete;
  AudioQualityAnalyzer(AudioQualityAnalyzer &&) = delete;
  AudioQualityAnalyzer &operator=(AudioQualityAnalyzer &&) = delete;

  void process_reference(const float *data, size_t size);
  void process_degraded(const float *data, size_t size);

  double get_pesq() const;
  double get_stoi() const;
  const std::vector<double> &get_spectrum() const;

private:
  struct FFTWPlanDeleter {
    void operator()(fftw_plan p) const;
  };
  struct FFTWDataDeleter {
    void operator()(void *p) const;
  };

  void apply_window(float *data) noexcept;
  void compute_spectral_envelope() noexcept;
  void time_align_signals() noexcept;

  void calculate_pesq();
  void calculate_stoi();

  const int sample_rate_;
  const int frame_size_;
  const int spectrum_bins_;

  std::unique_ptr<fftw_plan_s, FFTWPlanDeleter> fft_plan_;
  std::unique_ptr<double, FFTWDataDeleter> fft_in_;
  std::unique_ptr<fftw_complex, FFTWDataDeleter> fft_out_;

  std::vector<float> reference_;
  std::vector<float> degraded_;
  std::vector<double> spectrum_;
  std::vector<float> window_;

  double pesq_score_ = 0.0;
  double stoi_score_ = 0.0;

  bool reference_loaded_ = false;
  bool degraded_loaded_ = false;
};

} // namespace audio_quality
