#pragma once
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#if defined(__AVX__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define RTE_USE_AVX
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define RTE_USE_NEON
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace AudioEngine {

constexpr size_t MAX_TRACKS = 64;
constexpr size_t SIMD_ALIGNMENT = 32;
constexpr size_t FRAME_SIZE = 256;
constexpr float SPATIAL_DECAY = 0.5f;

struct SpatialParams {
  float azimuth = 0.0f;
  float elevation = 0.0f;
  float distance = 1.0f;
};

// Aligned allocator for SIMD
template <typename T, size_t Alignment> struct aligned_allocator {
  using value_type = T;
  T *allocate(size_t n);
  void deallocate(T *p, size_t) noexcept;
};

class AudioTrack {
public:
  AudioTrack();
  ~AudioTrack() = default;

  void set_buffer(const float *data, size_t size);
  void set_spatial_params(const SpatialParams &params);
  void play();
  void stop();

  bool is_active() const { return active.load(std::memory_order_relaxed); }
  void process_block(float *output, size_t frames);

private:
  std::vector<float> buffer;
  std::atomic<size_t> read_position{0};
  std::atomic_bool active{false};
  SpatialParams spatial_params;
  std::atomic_bool params_dirty{false};
  std::array<float, 2> current_panning{1.0f, 1.0f};

  void update_spatial_coeffs();
};

class RealTimeMixingEngine {
public:
  RealTimeMixingEngine();
  ~RealTimeMixingEngine();

  void initialize();
  void process(float *output, size_t frames);

  AudioTrack *add_track();
  void remove_track(AudioTrack *track);

private:
  std::array<std::unique_ptr<AudioTrack>, MAX_TRACKS> tracks;
  alignas(32) std::array<float, FRAME_SIZE * 2> temp_buffer;

  void simd_mix_buffers(float *__restrict dest, const float *__restrict src,
                        size_t size);
};

} // namespace AudioEngine
