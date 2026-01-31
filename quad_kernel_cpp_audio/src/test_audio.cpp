#include <iostream>
#include <memory>
#include <vector>

#include "aac_lc_he_aac_encoder.h"
#include "audio_buffer_ringbuffer.h"
#include "audio_effect_processor.h"
#include "audio_format_converter.h"
#include "audio_preprocessing_chain.h"
#include "audio_quality_analyzer.h"
#include "av_sync_controller.h"
#include "dynamic_range_processor.h"
#include "echo_cancellation_advanced.h"
#include "loudness_normalization.h"
#include "multichannel_upmix_engine.h"
#include "music_frequency_optimizer.h"
#include "noise_suppression_dsp.h"
#include "opus_encoder_advanced.h"
#include "podcast_speech_enhancer.h"
#include "real_time_mixing_engine.h"
#include "sample_rate_converter.h"
#include "spatial_audio_engine.h"
#include "surround_sound_processor.h"
#include "voice_activity_detection.h"

int main() {
  std::cout << "Initializing Quad Audio Components..." << std::endl;

  try {
    // 1. Audio Format Converter
    audio::AudioParams params{audio::Format::PCM, 48000, 2, 16};
    audio::LockFreeBufferPool pool(1024, 4);
    audio::PCMConverter pcm_conv(params, params, pool);
    std::cout << "[OK] AudioFormatConverter" << std::endl;

    // 2. Sample Rate Converter
    audio::SampleRateConverter sr_conv(44100, 48000, 1);
    std::cout << "[OK] SampleRateConverter" << std::endl;

    // 3. Audio Ring Buffer
    audio::AudioRingBuffer<float, 1024> ring_buf;
    std::cout << "[OK] AudioRingBuffer" << std::endl;

    // 4. AAC Encoder
    aac::EncoderConfig aac_cfg{48000, 2, 128000, false, false};
    aac::AACEncoder aac_enc(aac_cfg);
    std::cout << "[OK] AACEncoder" << std::endl;

    // 5. Opus Encoder
    OpusEncoderAdvanced opus_enc(48000, 2);
    std::cout << "[OK] OpusEncoder" << std::endl;

    // 6. Noise Suppression
    dsp::NoiseSuppressor::Config ns_cfg;
    dsp::NoiseSuppressor ns(ns_cfg);
    std::cout << "[OK] NoiseSuppressor" << std::endl;

    // 7. Audio Preprocessing Chain
    audio_processing::ProcessingChain chain;
    chain.add_processor(std::make_unique<audio_processing::Normalizer>(-1.0f));
    std::cout << "[OK] AudioPreprocessingChain" << std::endl;

    // 8. Audio Effect Processor
    dsp::AudioEffectProcessor effect_proc(512, 48000);
    std::cout << "[OK] AudioEffectProcessor" << std::endl;

    // 9. Spatial Audio Engine
    audio::SpatialAudioEngine spatial_engine(4, 512, 512);
    std::cout << "[OK] SpatialAudioEngine" << std::endl;

    // 10. Surround Sound Processor
    dsp::SurroundSoundProcessor surround_proc(dsp::ChannelLayout::LAYOUT_51);
    std::cout << "[OK] SurroundSoundProcessor" << std::endl;

    // 11. Multichannel Upmix Engine
    MultichannelUpmixEngine upmix_engine(6, 48000, UpmixMethod::Matrix);
    std::cout << "[OK] MultichannelUpmixEngine" << std::endl;

    // 12. Dynamic Range Processor
    DynamicRangeProcessor drp(-12.0f, 4.0f, 10.0f, 100.0f, 48000.0f,
                              DynamicRangeProcessor::Mode::COMPRESSOR);
    std::cout << "[OK] DynamicRangeProcessor" << std::endl;

    // 13. Loudness Normalization
    LoudnessNormalizer loudness_norm(48000);
    std::cout << "[OK] LoudnessNormalization" << std::endl;

    // 14. Echo Cancellation
    audio::EchoCancellerAdvanced aec(48000, 512, 1);
    std::cout << "[OK] EchoCancellationAdvanced" << std::endl;

    // 15. Audio Quality Analyzer
    audio_quality::AudioQualityAnalyzer aqa(48000, 512);
    std::cout << "[OK] AudioQualityAnalyzer" << std::endl;

    // 16. AV Sync Controller
    AvSyncController av_sync(48000, 100);
    std::cout << "[OK] AvSyncController" << std::endl;

    // 17. Music Frequency Optimizer
    dsp::MusicFrequencyOptimizer mfo(48000, 512);
    std::cout << "[OK] MusicFrequencyOptimizer" << std::endl;

    // 18. Podcast Speech Enhancer
    PodcastDSP::SpeechEnhancer pse(48000, 512);
    std::cout << "[OK] PodcastSpeechEnhancer" << std::endl;

    // 19. Voice Activity Detection
    VADConfig vad_cfg{48000, 10, 1.0f, 0.999f, 5};
    VoiceActivityDetector vad(vad_cfg);
    std::cout << "[OK] VoiceActivityDetection" << std::endl;

    // 20. Real-Time Mixing Engine
    AudioEngine::RealTimeMixingEngine mixing_engine;
    mixing_engine.initialize();
    std::cout << "[OK] RealTimeMixingEngine" << std::endl;

    std::cout << "\nAll components initialized successfully!" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "\nInitialization failed: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
