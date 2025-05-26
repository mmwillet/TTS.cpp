#ifndef GGML_TTS_EXT_H
#define GGML_TTS_EXT_H

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// TTS-specific GGML extensions
// These are the minimal functions needed from mmwillet's fork

// Modulo operation: a % b
GGML_API struct ggml_tensor *ggml_mod(struct ggml_context *ctx,
                                      struct ggml_tensor *a, float b);

// Cumulative sum along the first dimension
GGML_API struct ggml_tensor *ggml_cumsum(struct ggml_context *ctx,
                                         struct ggml_tensor *a);

// Linear upscaling (repeat elements)
GGML_API struct ggml_tensor *ggml_upscale_linear(struct ggml_context *ctx,
                                                 struct ggml_tensor *a,
                                                 int scale_factor);

// Round to nearest integer
GGML_API struct ggml_tensor *ggml_round(struct ggml_context *ctx,
                                        struct ggml_tensor *a);

// STFT with 7 parameters to match kokoro_model.cpp usage
struct ggml_tensor *stft(struct ggml_context *ctx, struct ggml_tensor *a,
                         struct ggml_tensor *window, int filter_length,
                         int hop_length, bool compute_abs_and_angle,
                         bool center);

// ISTFT with 8 parameters to match kokoro_model.cpp usage
struct ggml_tensor *istft(struct ggml_context *ctx, struct ggml_tensor *a,
                          struct ggml_tensor *window_squared_sum,
                          struct ggml_tensor *window, int filter_length,
                          int hop_length, bool from_abs_and_angle, bool center);

#ifdef __cplusplus
}
#endif

#endif // GGML_TTS_EXT_H