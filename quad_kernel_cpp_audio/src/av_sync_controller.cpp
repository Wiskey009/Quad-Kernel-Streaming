#include "av_sync_controller.h"
#include <algorithm>
#include <deque>

AvSyncController::AvSyncController(size_t audio_buffer_size,
                                   size_t /*max_history_ms*/)
    : stretch_buffer_(audio_buffer_size * 2) {
  for (int i = 0; i < 3; ++i) {
    auto buf = std::make_unique<AudioBuffer>();
    buf->data.resize(audio_buffer_size * 2);
    audio_buffers_.push_back(std::move(buf));
  }
}

AvSyncController::~AvSyncController() = default;

void AvSyncController::feed_audio(const float *data, size_t samples,
                                  int64_t pts) {
  size_t idx = free_index_.load(std::memory_order_acquire);
  AudioBuffer *buf = audio_buffers_[idx].get();

  if (buf->head + samples > buf->data.size())
    buf->head = 0;
  std::copy_n(data, samples, buf->data.data() + buf->head);
  buf->head += samples;
  buf->valid.store(true, std::memory_order_release);

  current_pts_.store(pts);
  update_drift_compensation(pts, get_current_pts());

  if (resampling_required_.load(std::memory_order_acquire)) {
    apply_time_stretch(1.0f + static_cast<float>(drift_ppm_.load() / 1e6));
  }

  ready_index_.store(idx, std::memory_order_release);
  free_index_.store((idx + 1) % 3, std::memory_order_release);
}

void AvSyncController::feed_video(int64_t pts, bool key_frame) {
  video_queue_.push_back({pts, key_frame});
  if (video_queue_.size() > 60)
    video_queue_.pop_front();
}

AvSyncController::AudioFrame AvSyncController::get_corrected_audio() {
  size_t idx = ready_index_.load(std::memory_order_acquire);
  AudioBuffer *buf = audio_buffers_[idx].get();
  if (!buf->valid.load(std::memory_order_acquire))
    return {nullptr, 0, 0};

  size_t len = buf->head - buf->tail;
  float *ptr = buf->data.data() + buf->tail;
  buf->tail = buf->head; // Mark as read
  buf->valid.store(false, std::memory_order_release);

  return {ptr, len, current_pts_.load()};
}

int64_t AvSyncController::get_current_pts() const {
  return !video_queue_.empty() ? video_queue_.back().pts : current_pts_.load();
}

void AvSyncController::update_drift_compensation(int64_t audio_pts,
                                                 int64_t video_pts) {
  double error = static_cast<double>(audio_pts - video_pts);
  pid_.integral += error;
  double derivative = error - pid_.prev_error;
  double correction =
      pid_.Kp * error + pid_.Ki * pid_.integral + pid_.Kd * derivative;
  drift_ppm_.store(correction);
  pid_.prev_error = error;
  resampling_required_.store(std::abs(correction) > 50.0);
}

void AvSyncController::apply_time_stretch(float factor) {
  AudioBuffer *buf = audio_buffers_[ready_index_.load()].get();
  size_t samples = buf->head - buf->tail;
  if (samples == 0)
    return;

  process_audio_simd(buf->data.data() + buf->tail, stretch_buffer_.data(),
                     samples, factor);
  std::copy_n(stretch_buffer_.data(), samples, buf->data.data() + buf->tail);
}

void AvSyncController::process_audio_simd(const float *src, float *dest,
                                          size_t samples,
                                          float stretch_factor) {
  // Basic linear interpolation fallback
  for (size_t i = 0; i < samples; ++i) {
    float pos = i * stretch_factor;
    size_t idx = static_cast<size_t>(pos);
    float frac = pos - idx;
    if (idx + 1 < samples) {
      dest[i] = src[idx] * (1.0f - frac) + src[idx + 1] * frac;
    } else {
      dest[i] = src[idx % samples];
    }
  }
}
