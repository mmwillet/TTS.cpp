#ifndef util_h
#define util_h

#include <functional>
#include <math.h>
#include <random>
#include <stdio.h>
#include <string>
#include <vector>
#include <stdint.h>
#include <sys/types.h>
#include "ggml-metal.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpp.h"

#define TTS_ABORT(...) tts_abort(__FILE__, __LINE__, __VA_ARGS__)

struct model_tensor_meta {
	uint32_t n_tensors = 0;
	size_t n_bytes = 0;
};

void random_gen(int count, float * tgt, float min = 0.0f, float max = 1.0);

std::pair<int, std::string> parse_layer_count(std::string name, int skip = 0);

struct model_tensor_meta compute_tensor_meta(std::string name_prefix, ggml_context * weight_ctx, std::function<void(ggml_tensor*)>* callback = nullptr);
struct ggml_tensor * snake_1d(ggml_context * ctx, struct ggml_tensor * alpha, struct ggml_tensor * a);
uint64_t get_cpu_count();
int search_for_gguf_keys(gguf_context * meta, std::vector<std::string> possible_keys);

// a simple window function for stft
void hann_window(size_t n_fft, float * tgt);

// currently this assumes a center view in which the output vector is reflectively padded by n_fft / 2 on each side.
void compute_window_squared_sum(size_t n_fft, size_t hop, size_t n_frames, float * tgt, float * window);

// these functions wrap the stft and istft ggml ops and compute the necessary view and division ops for their indepentent settings.
struct ggml_tensor * stft(ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * window, size_t n_fft, size_t hop, bool abs_and_angle, bool one_sided);
struct ggml_tensor * istft(ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * window_squared_sum, struct ggml_tensor * window, size_t n_fft, size_t hop, bool abs_and_angle, bool one_sided);

// This is a custom ops for sine_generation in the kokoro model.
void uv_noise_compute(struct ggml_tensor * dst, const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata);

bool has_suffix(std::string value, std::string suffix);
bool has_prefix(std::string value, std::string prefix);

std::string replace_any(std::string target, std::string to_replace, std::string replacement);

void tts_abort(const char * file, int line, const char * fmt, ...);

#endif
