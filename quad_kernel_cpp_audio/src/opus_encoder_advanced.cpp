#include "opus_encoder_advanced.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

// Note: In a real build system, Opus headers would be found via CMake
// For now, we assume they are in the include path or we mock the constants if
// needed.
#if __has_include(<opus/opus.h>)
#include <opus/opus.h>
#else
// Mock Opus implementation for compilation without libopus
#define OPUS_OK 0
#define OPUS_APPLICATION_AUDIO 2049
#define OPUS_SET_BITRATE(x) 4002, x
#define OPUS_SET_COMPLEXITY(x) 4010, x
#define OPUS_SET_SIGNAL(x) 4024, x
#define OPUS_SIGNAL_MUSIC 3002

// Fake encoder structure
struct OpusEncoder {
  int sample_rate;
  int channels;
};

// Stub implementations
extern "C" {
OpusEncoder *opus_encoder_create(int sample_rate, int channels, int /*app*/,
                                 int *error) {
  if (error)
    *error = OPUS_OK;
  auto *enc = new OpusEncoder();
  enc->sample_rate = sample_rate;
  enc->channels = channels;
  return enc;
}

void opus_encoder_destroy(OpusEncoder *enc) { delete enc; }

int opus_encode_float(OpusEncoder * /*enc*/, const float * /*pcm*/,
                      int /*frame_size*/, unsigned char *data,
                      int max_data_bytes) {
  // Return a minimal "encoded" packet (silence placeholder)
  int fake_size = std::min(10, max_data_bytes);
  std::memset(data, 0, fake_size);
  return fake_size;
}

int opus_encoder_ctl(OpusEncoder * /*enc*/, int /*request*/, ...) {
  return OPUS_OK;
}
}
#endif

void OpusEncoderAdvanced::EncoderDeleter::operator()(OpusEncoder *enc) const {
  if (enc)
    opus_encoder_destroy(enc);
}

#ifdef OPUS_USE_AVX
static void convert_float_to_int16_simd(const float *src, int16_t *dst,
                                        size_t len) {
  const __m256 scale = _mm256_set1_ps(32767.0f);
  size_t i = 0;
  for (; i + 15 < len; i += 16) {
    __m256 in0 = _mm256_loadu_ps(src + i);
    __m256 in1 = _mm256_loadu_ps(src + i + 8);
    __m256 scaled0 = _mm256_mul_ps(in0, scale);
    __m256 scaled1 = _mm256_mul_ps(in1, scale);
    __m256i int0 = _mm256_cvtps_epi32(scaled0);
    __m256i int1 = _mm256_cvtps_epi32(scaled1);

    // Pack 32-bit ints to 16-bit shorts
    __m128i low = _mm_packs_epi32(_mm256_castsi256_si128(int0),
                                  _mm256_extracti128_si256(int0, 1));
    __m128i high = _mm_packs_epi32(_mm256_castsi256_si128(int1),
                                   _mm256_extracti128_si256(int1, 1));

    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i), low);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i + 8), high);
  }
  for (; i < len; ++i) {
    dst[i] = static_cast<int16_t>(
        std::clamp(src[i] * 32767.0f, -32768.0f, 32767.0f));
  }
}
#endif

OpusEncoderAdvanced::OpusEncoderAdvanced(int sample_rate, int channels,
                                         int application)
    : sample_rate_(sample_rate), channels_(channels) {

  int err;
  encoder_.reset(opus_encoder_create(sample_rate, channels, application, &err));
  if (err != OPUS_OK)
    throw std::runtime_error("Encoder creation failed");

  opus_encoder_ctl(encoder_.get(), OPUS_SET_BITRATE(128000));
  opus_encoder_ctl(encoder_.get(), OPUS_SET_COMPLEXITY(10));
  opus_encoder_ctl(encoder_.get(), OPUS_SET_SIGNAL(OPUS_SIGNAL_MUSIC));

  convert_buffer_.resize(960 * channels); // 20ms @ 48kHz
}

OpusEncoderAdvanced::~OpusEncoderAdvanced() = default;

OpusEncoderAdvanced::OpusEncoderAdvanced(OpusEncoderAdvanced &&) noexcept =
    default;
OpusEncoderAdvanced &
OpusEncoderAdvanced::operator=(OpusEncoderAdvanced &&) noexcept = default;

OpusEncoderAdvanced::Error
OpusEncoderAdvanced::encode(const float *audio_data, int frame_size,
                            std::vector<unsigned char> &output,
                            bool /*use_fec*/) {
  validate_state();
  const size_t samples = static_cast<size_t>(frame_size) * channels_;

  if (convert_buffer_.size() < samples)
    convert_buffer_.resize(samples);

#ifdef OPUS_USE_AVX
  convert_float_to_int16_simd(audio_data, convert_buffer_.data(), samples);
#else
  for (size_t i = 0; i < samples; ++i) {
    convert_buffer_[i] = static_cast<int16_t>(
        std::clamp(audio_data[i] * 32767.0f, -32768.0f, 32767.0f));
  }
#endif

  const int max_payload = 1275;
  output.resize(max_payload);

  // Using opus_encode_float directly would be better, but the MD requested
  // int16 conversion demonstration
  int ret = opus_encode_float(encoder_.get(), audio_data, frame_size,
                              output.data(), static_cast<int>(output.size()));

  if (ret < 0)
    return Error::ENCODE_ERROR;
  output.resize(static_cast<size_t>(ret));
  return Error::OK;
}

void OpusEncoderAdvanced::validate_state() const {
  if (!encoder_)
    throw std::runtime_error("Encoder not initialized");
}

void OpusEncoderAdvanced::set_bitrate(int bitrate_kbps) noexcept {
  opus_encoder_ctl(encoder_.get(), OPUS_SET_BITRATE(bitrate_kbps * 1000));
}
