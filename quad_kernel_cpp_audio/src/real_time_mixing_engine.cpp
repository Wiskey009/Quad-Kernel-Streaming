#include "real_time_mixing_engine.h"
#include <algorithm>
#include <cstring>

// Allocator removed - using standard std::vector allocator

namespace AudioEngine {

AudioTrack::AudioTrack() {}

void AudioTrack::set_buffer(const float *data, size_t size) {
  buffer.assign(data, data + size);
}

void AudioTrack::play() {
  read_position.store(0, std::memory_order_relaxed);
  active.store(true, std::memory_order_release);
}

void AudioTrack::stop() { active.store(false, std::memory_order_release); }

void AudioTrack::set_spatial_params(const SpatialParams &params) {
  spatial_params = params;
  params_dirty.store(true, std::memory_order_release);
}

void AudioTrack::update_spatial_coeffs() {
  const float rad = spatial_params.azimuth * (float)M_PI / 180.0f;
  const float dist = std::max(spatial_params.distance, 0.01f);
  const float atten = SPATIAL_DECAY / dist;
  current_panning[0] = atten * (0.5f * cosf(rad) + 0.5f);
  current_panning[1] = atten * (0.5f * -cosf(rad) + 0.5f);
  params_dirty.store(false, std::memory_order_release);
}

void AudioTrack::process_block(float *output, size_t frames) {
  if (!active.load(std::memory_order_acquire) || buffer.empty())
    return;
  if (params_dirty.load(std::memory_order_acquire))
    update_spatial_coeffs();

  size_t pos = read_position.load(std::memory_order_relaxed);
  for (size_t i = 0; i < frames; ++i) {
    float sample = buffer[pos];
    output[i * 2] += sample * current_panning[0];
    output[i * 2 + 1] += sample * current_panning[1];
    pos = (pos + 1) % buffer.size();
  }
  read_position.store(pos, std::memory_order_relaxed);
}

RealTimeMixingEngine::RealTimeMixingEngine() {
  for (auto &t : tracks)
    t = std::unique_ptr<AudioTrack>(new AudioTrack());
}

RealTimeMixingEngine::~RealTimeMixingEngine() = default;

void RealTimeMixingEngine::initialize() { temp_buffer.fill(0.0f); }

void RealTimeMixingEngine::process(float *output, size_t frames) {
  std::memset(output, 0, frames * 2 * sizeof(float));
  for (auto &t : tracks) {
    if (!t->is_active())
      continue;
    temp_buffer.fill(0.0f);
    t->process_block(temp_buffer.data(), frames);
    simd_mix_buffers(output, temp_buffer.data(), frames * 2);
  }
}

void RealTimeMixingEngine::simd_mix_buffers(float *__restrict dest,
                                            const float *__restrict src,
                                            size_t size) {
  size_t i = 0;
#ifdef RTE_USE_AVX
  for (; i + 7 < size; i += 8) {
    __m256 v_d = _mm256_loadu_ps(dest + i);
    __m256 v_s = _mm256_loadu_ps(src + i);
    _mm256_storeu_ps(dest + i, _mm256_add_ps(v_d, v_s));
  }
#endif
  for (; i < size; ++i)
    dest[i] += src[i];
}

AudioTrack *RealTimeMixingEngine::add_track() {
  for (auto &t : tracks)
    if (!t->is_active())
      return t.get();
  return nullptr;
}

void RealTimeMixingEngine::remove_track(AudioTrack *track) {
  if (track)
    track->stop();
}

} // namespace AudioEngine
