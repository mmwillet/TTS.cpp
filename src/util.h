#ifndef util_h
#define util_h

#include <cassert>
#include <cstring>
#include <functional>
#include <math.h>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <sys/types.h>
#include <vector>

// Direct llama.cpp GGML headers
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml.h"
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
// Removed ggml-tts.h - using standard llama.cpp/ggml operations
#include "gguf.h"

#define TTS_ABORT(...) tts_abort(__FILE__, __LINE__, __VA_ARGS__)
#define TTS_ASSERT(x)                                                          \
  if (!(x))                                                                    \
  TTS_ABORT("TTS_ASSERT(%s) failed", #x)

struct model_tensor_meta {
  uint32_t n_tensors = 0;
  size_t n_bytes = 0;
};

void random_gen(int count, float *tgt, float min = 0.0f, float max = 1.0);

std::pair<int, std::string> parse_layer_count(std::string name, int skip = 0);

struct model_tensor_meta
compute_tensor_meta(std::string name_prefix, ggml_context *weight_ctx,
                    std::function<void(ggml_tensor *)> *callback = nullptr);
// Snake activation function using standard GGML operations
struct ggml_tensor *snake_1d(ggml_context *ctx, struct ggml_tensor *alpha,
                             struct ggml_tensor *a);
int search_for_gguf_keys(gguf_context *meta,
                         std::vector<std::string> possible_keys);

// a simple window function for stft
void hann_window(size_t n_fft, std::vector<float> &tgt);

// currently this assumes a center view in which the output vector is
// reflectively padded by n_fft / 2 on each side.
void compute_window_squared_sum(size_t n_fft, size_t hop, size_t n_frames,
                                float *tgt, float *window);

// STFT/ISTFT functions removed - they were using custom GGML operations
// that don't exist in standard llama.cpp/ggml. If needed, they should be
// implemented using standard GGML operations or as custom map operations.

// This is a custom op for sine_generation in the Kokoro model.
void uv_noise_compute(struct ggml_tensor *dst, const struct ggml_tensor *a,
                      const struct ggml_tensor *b, const struct ggml_tensor *c,
                      int ith, int nth, void *userdata);

// This is a custom op for logit correction in the Dia model.
void cfg_scale(struct ggml_tensor *dst, const struct ggml_tensor *a,
               const struct ggml_tensor *b, int ith, int nth, void *userdata);

bool has_suffix(std::string value, std::string suffix);
bool has_prefix(std::string value, std::string prefix);

std::vector<std::string> split(std::string target, std::string split_on,
                               bool include_split_characters = false);
std::vector<std::string> split(std::string target, const char split_on,
                               bool include_split_characters = false);
std::string strip(std::string target, std::string vals = " ");
std::string replace_any(std::string target, std::string to_replace,
                        std::string replacement);

[[noreturn]] void tts_abort(const char *file, int line, const char *fmt, ...);

#endif
