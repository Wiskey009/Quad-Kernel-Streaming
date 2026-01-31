#include "spatial_audio_engine.h"
#include <algorithm>
#include <cmath>

namespace audio {

HRTFDatabase::HRTFDatabase(size_t max_frames, size_t fft_size)
    : max_frames_(max_frames), fft_size_(fft_size) {
  left_ir_.resize(max_frames * fft_size);
  right_ir_.resize(max_frames * fft_size);
}

void HRTFDatabase::load_frame(size_t index, const float *left,
                              const float *right) {
  if (index >= max_frames_)
    return;
  std::copy_n(left, fft_size_, left_ir_.data() + index * fft_size_);
  std::copy_n(right, fft_size_, right_ir_.data() + index * fft_size_);
}

void HRTFDatabase::get_interpolated_frame(float az, float el, float *left,
                                          float *right) const {
  // Basic mapping: azimuth to frame index
  float deg = az * 180.0f / (float)M_PI;
  if (deg < 0)
    deg += 360.0f;
  size_t frame_idx = (static_cast<size_t>(deg)) % max_frames_;

  const float *base_l = left_ir_.data() + frame_idx * fft_size_;
  const float *base_r = right_ir_.data() + frame_idx * fft_size_;
  std::copy_n(base_l, fft_size_, left);
  std::copy_n(base_r, fft_size_, right);
}

SpatialAudioSource::SpatialAudioSource(size_t /*buffer_frames*/,
                                       size_t fft_size)
    : fft_size_(fft_size) {
  prev_l_out_.resize(fft_size, 0.0f);
  prev_r_out_.resize(fft_size, 0.0f);
}

void SpatialAudioSource::process_block(const AudioBlock &block,
                                       const HRTFDatabase &hrtf,
                                       const SpatialParams &params,
                                       float *out_l, float *out_r) {
  for (size_t i = 0; i < block.samples; ++i) {
    input_queue_.push(block.data[i]);
  }

  while (input_queue_.size() >= fft_size_) {
    process_convolution(hrtf, params);
  }

  // Overlap-add
  for (size_t i = 0; i < fft_size_; ++i) {
    out_l[i] += prev_l_out_[i];
    out_r[i] += prev_r_out_[i];
  }
}

void SpatialAudioSource::process_convolution(const HRTFDatabase &hrtf,
                                             const SpatialParams &params) {
  std::vector<float> input(fft_size_);
  std::vector<float> hrtf_l(fft_size_), hrtf_r(fft_size_);

  for (size_t i = 0; i < fft_size_; ++i)
    input_queue_.pop(input[i]);
  hrtf.get_interpolated_frame(params.azimuth, params.elevation, hrtf_l.data(),
                              hrtf_r.data());

  // Frequency domain convolution should be here, simplified to time domain for
  // demo
  for (size_t i = 0; i < fft_size_; ++i) {
    prev_l_out_[i] = input[i] * hrtf_l[i];
    prev_r_out_[i] = input[i] * hrtf_r[i];
  }

  float gain = 1.0f / (params.distance + 0.1f);
  for (size_t i = 0; i < fft_size_; ++i) {
    prev_l_out_[i] *= gain;
    prev_r_out_[i] *= gain;
  }
}

SpatialAudioEngine::SpatialAudioEngine(size_t num_sources, size_t block_size,
                                       size_t fft_size)
    : block_size_(block_size), fft_size_(fft_size) {
  for (size_t i = 0; i < num_sources; ++i) {
    sources_.push_back(
        std::make_unique<SpatialAudioSource>(block_size * 4, fft_size));
  }
  accum_l_.resize(fft_size, 0.0f);
  accum_r_.resize(fft_size, 0.0f);
}

void SpatialAudioEngine::process(const AudioBlock *inputs,
                                 const SpatialParams *params, size_t num_blocks,
                                 float *output_l, float *output_r) {
  std::fill(accum_l_.begin(), accum_l_.end(), 0.0f);
  std::fill(accum_r_.begin(), accum_r_.end(), 0.0f);

  for (size_t i = 0; i < num_blocks; ++i) {
    sources_[i]->process_block(inputs[i], *hrtf_db_, params[i], accum_l_.data(),
                               accum_r_.data());
  }

  for (size_t i = 0; i < fft_size_; ++i) {
    output_l[i] = accum_l_[i];
    output_r[i] = accum_r_[i];
  }
}

void SpatialAudioEngine::set_hrtf_database(std::unique_ptr<HRTFDatabase> db) {
  hrtf_db_ = std::move(db);
}

} // namespace audio
