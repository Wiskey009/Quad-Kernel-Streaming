#pragma once
#include <atomic>
#include <cmath>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#if defined(__AVX__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define ASC_USE_AVX
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define ASC_USE_NEON
#endif

class AvSyncController {
public:
  struct AudioFrame {
    float *data;
    size_t samples;
    int64_t pts;
  };
  struct VideoFrame {
    int64_t pts;
    bool key_frame;
  };

  AvSyncController(size_t audio_buffer_size, size_t max_history_ms);
  ~AvSyncController();

  void feed_audio(const float *data, size_t samples, int64_t pts);
  void feed_video(int64_t pts, bool key_frame);
  AudioFrame get_corrected_audio();
  int64_t get_current_pts() const;

private:
  struct AudioBuffer {
    std::vector<float> data;
    size_t head = 0;
    size_t tail = 0;
    std::atomic<bool> valid{false};
  };

  void apply_time_stretch(float factor);
  void process_audio_simd(const float *src, float *dest, size_t samples,
                          float stretch_factor);
  void update_drift_compensation(int64_t audio_pts, int64_t video_pts);

  struct PIDController {
    double Kp = 0.05, Ki = 0.001, Kd = 0.01;
    double integral = 0.0, prev_error = 0.0;
    int64_t last_update = 0;
  };

  std::vector<std::unique_ptr<AudioBuffer>> audio_buffers_;
  std::atomic<size_t> ready_index_{0};
  std::atomic<size_t> free_index_{1};
  std::deque<VideoFrame> video_queue_;

  PIDController pid_;
  std::atomic<int64_t> current_pts_{0};
  std::atomic<double> drift_ppm_{0.0};
  std::atomic<bool> resampling_required_{false};

  alignas(32) std::vector<float> stretch_buffer_;
};
