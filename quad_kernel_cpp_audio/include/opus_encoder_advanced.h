#pragma once
#include <cstdint>
#include <memory>
#include <vector>


#if defined(__AVX__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define OPUS_USE_AVX
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define OPUS_USE_NEON
#endif

// Forward declaration of Opus types to avoid header dependency in public header
// if possible
struct OpusEncoder;

class OpusEncoderAdvanced {
public:
  enum class Error { OK, BUFFER_TOO_SMALL, INVALID_STATE, ENCODE_ERROR };

  OpusEncoderAdvanced(int sample_rate, int channels,
                      int application = 2049 /* OPUS_APPLICATION_AUDIO */);
  ~OpusEncoderAdvanced();

  OpusEncoderAdvanced(const OpusEncoderAdvanced &) = delete;
  OpusEncoderAdvanced &operator=(const OpusEncoderAdvanced &) = delete;
  OpusEncoderAdvanced(OpusEncoderAdvanced &&) noexcept;
  OpusEncoderAdvanced &operator=(OpusEncoderAdvanced &&) noexcept;

  Error encode(const float *audio_data, int frame_size,
               std::vector<unsigned char> &output, bool use_fec = false);

  int get_bitrate() const noexcept;
  void set_bitrate(int bitrate_kbps) noexcept;
  void enable_dtx(bool enable) noexcept;

private:
  struct EncoderDeleter {
    void operator()(OpusEncoder *enc) const;
  };

  std::unique_ptr<OpusEncoder, EncoderDeleter> encoder_;
  int sample_rate_;
  int channels_;
  std::vector<float> resample_buffer_;
  std::vector<int16_t> convert_buffer_;

  void simd_resample_if_needed(const float *input, int frames);
  void validate_state() const;
};
