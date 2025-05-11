#pragma once

#include <functional>
#include <math.h>
#include <random>
#include <stdio.h>
#include <string>
#include <cstring>
#include <vector>
#include <stdint.h>
#include <sys/types.h>
#include "common.h"
#include "ggml-metal.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpp.h"

#define TTS_ABORT(...) ggml_abort(__FILE__, __LINE__, __VA_ARGS__)
#define TTS_ASSERT(x) if (!(x)) TTS_ABORT("TTS_ASSERT(%s) failed", #x)

struct model_tensor_meta {
    explicit model_tensor_meta(ggml_context * weight_ctx, sv name_prefix,
                               uint8_t size_multiplier_percent = 140, uint32_t dedicated_add_on_size = 0);

    uint32_t n_tensors;
    size_t mem_size;
    size_t buf_size;
};

void random_gen(int count, float * tgt, float min = 0.0f, float max = 1.0);

pair<uint32_t, sv> parse_layer_count(sv name);

ggml_tensor * snake_1d(ggml_context * ctx, ggml_tensor * alpha, ggml_tensor * a);

// a simple window function for stft
void hann_window(size_t n_fft, vector<float>& tgt);

// currently this assumes a center view in which the output vector is reflectively padded by n_fft / 2 on each side.
void compute_window_squared_sum(size_t n_fft, size_t hop, size_t n_frames, float * tgt, float * window);

// these functions wrap the stft and istft ggml ops and compute the necessary view and division ops for their indepentent settings.
ggml_tensor * stft(ggml_context * ctx, ggml_tensor * a, ggml_tensor * window, size_t n_fft, size_t hop, bool abs_and_angle, bool one_sided);
ggml_tensor * istft(ggml_context * ctx, ggml_tensor * a, ggml_tensor * window_squared_sum, ggml_tensor * window, size_t n_fft, size_t hop, bool abs_and_angle, bool one_sided);

// This is a custom op for sine_generation in the Kokoro model.
void uv_noise_compute(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c, int ith, int nth, void * userdata);

// This is a custom op for logit correction in the Dia model.
void cfg_scale(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * userdata);

string strip(const string & target, const string & vals = " ");

inline ptrdiff_t binary_search_idx(const auto & haystack, const auto & needle, auto cmp) {
    const auto lt{[cmp](auto x, auto y){ return cmp(x, y) == weak_ordering::less; }};
    const auto eq{[cmp](auto x, auto y){ return cmp(x, y) == weak_ordering::equivalent; }};
    const auto result{lower_bound(cbegin(haystack), cend(haystack), needle, lt)};
    return result == cend(haystack) || !eq(*result, needle) ? ~0 : distance(cbegin(haystack), result);
}
#define pair_first_cmp_sv [](auto x, sv y){ return std::get<0>(x) <=> y; }

inline int sv_int(const sv & x) {
    int result{};
    const errc err{from_chars(cbegin(x), cend(x), result).ec};
    TTS_ASSERT(err == errc{});
    return result;
}
