# real_time_mixing_engine



```cpp
// real_time_mixing_engine.hpp
#pragma once

#include <vector>
#include <array>
#include <atomic>
#include <memory>
#include <cmath>
#include <immintrin.h>  // AVX intrinsics
#include <type_traits>

namespace AudioEngine {

constexpr size_t MAX_TRACKS = 64;
constexpr size_t SIMD_ALIGNMENT = 32;
constexpr size_t FRAME_SIZE = 256;
constexpr float SPATIAL_DECAY = 0.5f;

struct SpatialParams {
    float azimuth = 0.0f;    // -180 to +180 degrees
    float elevation = 0.0f;  // -90 to +90 degrees
    float distance = 1.0f;   // Normalized 0.0-1.0
};

class AudioTrack {
public:
    AudioTrack();
    
    void set_buffer(const float* data, size_t size);
    void set_spatial_params(const SpatialParams& params);
    void play();
    void stop();
    
    bool is_active() const { return active.load(std::memory_order_relaxed); }
    void process_block(float* output, size_t frames);

private:
    std::vector<float, aligned_allocator<float, SIMD_ALIGNMENT>> buffer;
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
    void process(float* output, size_t frames);
    
    AudioTrack* add_track();
    void remove_track(AudioTrack* track);
    
private:
    std::array<std::unique_ptr<AudioTrack>, MAX_TRACKS> tracks;
    std::array<float, FRAME_SIZE * 2> temp_buffer;
    
    // SIMD optimized processing functions
    void simd_mix_buffers(float* __restrict dest, const float* __restrict src, size_t size);
    void apply_spatial_effects(float* __restrict output, 
                              const float* __restrict input, 
                              const std::array<float, 2>& coeffs, 
                              size_t frames);
};

// Custom aligned allocator for SIMD types
template <typename T, size_t Alignment>
struct aligned_allocator {
    using value_type = T;
    static_assert(Alignment >= alignof(T), "Alignment too small");

    T* allocate(size_t n) {
        return static_cast<T*>(aligned_alloc(Alignment, n * sizeof(T)));
    }

    void deallocate(T* p, size_t) noexcept {
        free(p);
    }
};

} // namespace AudioEngine
```

```cpp
// real_time_mixing_engine.cpp
#include "real_time_mixing_engine.hpp"

namespace AudioEngine {

// AudioTrack Implementation
AudioTrack::AudioTrack() : buffer(SIMD_ALIGNMENT * 4) {}

void AudioTrack::set_buffer(const float* data, size_t size) {
    buffer.assign(data, data + size);
}

void AudioTrack::set_spatial_params(const SpatialParams& params) {
    spatial_params = params;
    params_dirty.store(true, std::memory_order_release);
}

void AudioTrack::play() {
    read_position.store(0, std::memory_order_relaxed);
    active.store(true, std::memory_order_release);
}

void AudioTrack::stop() {
    active.store(false, std::memory_order_release);
}

void AudioTrack::update_spatial_coeffs() {
    // Simplified spatial panning model (replace with HRTF for advanced use)
    const float rad = spatial_params.azimuth * M_PI / 180.0f;
    const float distance_factor = 1.0f / std::max(spatial_params.distance, 0.01f);
    const float attenuation = SPATIAL_DECAY * distance_factor;
    
    current_panning[0] = attenuation * (0.5f * cosf(rad) + 0.5f);
    current_panning[1] = attenuation * (0.5f * -cosf(rad) + 0.5f);
    params_dirty.store(false, std::memory_order_release);
}

void AudioTrack::process_block(float* output, size_t frames) {
    if (!active.load(std::memory_order_acquire) || buffer.empty()) return;
    
    if (params_dirty.load(std::memory_order_acquire)) {
        update_spatial_coeffs();
    }

    size_t samples_processed = 0;
    size_t current_pos = read_position.load(std::memory_order_relaxed);
    const size_t buffer_size = buffer.size();

    while (samples_processed < frames) {
        const size_t samples_to_copy = std::min(frames - samples_processed, 
                                               buffer_size - current_pos);

        // Process stereo interleaved data
        for (size_t i = 0; i < samples_to_copy; ++i) {
            const size_t out_idx = (samples_processed + i) * 2;
            const float sample = buffer[current_pos + i];
            output[out_idx] += sample * current_panning[0];
            output[out_idx + 1] += sample * current_panning[1];
        }

        samples_processed += samples_to_copy;
        current_pos = (current_pos + samples_to_copy) % buffer_size;
    }

    read_position.store(current_pos, std::memory_order_relaxed);
}

// RealTimeMixingEngine Implementation
RealTimeMixingEngine::RealTimeMixingEngine() {
    for (auto& track : tracks) {
        track = std::make_unique<AudioTrack>();
    }
}

RealTimeMixingEngine::~RealTimeMixingEngine() = default;

void RealTimeMixingEngine::initialize() {
    temp_buffer.fill(0.0f);
}

void RealTimeMixingEngine::process(float* output, size_t frames) {
    std::fill_n(output, frames * 2, 0.0f);

    for (auto& track : tracks) {
        if (!track->is_active()) continue;
        
        std::fill(temp_buffer.begin(), temp_buffer.end(), 0.0f);
        track->process_block(temp_buffer.data(), frames);
        simd_mix_buffers(output, temp_buffer.data(), frames * 2);
    }
}

void RealTimeMixingEngine::simd_mix_buffers(float* __restrict dest, 
                                           const float* __restrict src, 
                                           size_t size) {
    size_t i = 0;
#if defined(__AVX__)
    const size_t simd_samples = size - (size % 8);
    for (; i < simd_samples; i += 8) {
        __m256 dest_vec = _mm256_load_ps(dest + i);
        __m256 src_vec = _mm256_load_ps(src + i);
        __m256 result = _mm256_add_ps(dest_vec, src_vec);
        _mm256_store_ps(dest + i, result);
    }
#elif defined(__ARM_NEON)
    const size_t simd_samples = size - (size % 4);
    for (; i < simd_samples; i += 4) {
        float32x4_t dest_vec = vld1q_f32(dest + i);
        float32x4_t src_vec = vld1q_f32(src + i);
        float32x4_t result = vaddq_f32(dest_vec, src_vec);
        vst1q_f32(dest + i, result);
    }
#endif

    // Process remaining samples
    for (; i < size; ++i) {
        dest[i] += src[i];
    }
}

AudioTrack* RealTimeMixingEngine::add_track() {
    for (auto& track : tracks) {
        if (!track->is_active()) {
            track->stop();
            return track.get();
        }
    }
    return nullptr;
}

void RealTimeMixingEngine::remove_track(AudioTrack* track) {
    if (track) {
        track->stop();
    }
}

} // namespace AudioEngine
```

## Integration Points
1. **Initialization**: Create `RealTimeMixingEngine` instance once during audio system initialization
2. **Audio I/O**: Call `process()` in real-time audio thread with output buffer
3. **Track Management**:
   - `add_track()` for new audio sources (returns `AudioTrack*`)
   - `remove_track()` when sound completes
4. **Spatial Control**: Use `AudioTrack::set_spatial_params()` for dynamic position updates
5. **Playback Control**: Use `play()`/`stop()` on tracks for state management

## Performance Notes
- **Lock-free**: Atomic operations for track state changes
- **SIMD Optimized**: AVX/Neon intrinsics for mixing operations
- **Real-time Safe**: No allocations in audio thread, fixed-size buffers
- **Cache-friendly**: Memory-aligned buffers and linear data access patterns
- **Multi-core Ready**: Track processing can be parallelized (future extension)