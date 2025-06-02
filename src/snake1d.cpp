#include "util.h"
#include <ranges>
#include <algorithm>

/**
 * Removes the last axis, for cases where it's redundantly of length 1.
 * assert x.ndim == 2; numpy.squeeze(x, axis=-1)
 */
static ggml_tensor * squeeze_2d_1d_e0(ggml_context * ctx, ggml_tensor * x) {
    TTS_ASSERT(x->ne[0] == 1 && x->ne[2] == 1 && x->ne[3] == 1);
    TTS_ASSERT(ggml_is_contiguous(x));
    return ggml_reshape_1d(ctx, x, x->ne[1]);
}

ggml_tensor * reciprocal(ggml_context * ctx, ggml_tensor * x) {
    TTS_ASSERT(x->ne[0] == 1);
    static constexpr float one = 1.0f;
    ggml_tensor * numerator = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, x->ne[1]);
    // stride trick so that the scalar numerator can be divided by x.
    numerator->nb[1] = 0;
    numerator->data = const_cast<float *>(&one);
    return ggml_div(ctx, numerator, x);
}

/*
 * ### Experiment results ###
 * The effect was NOT statistically significant.
 * Although optimizing ggml_sin had a measurable improvement, it seems not much was to be gained from optimizing here.
 * Perf data (-nt 1):
 * - New fused kernel <256> : 1426 samples, 397 own samples
 * - New fused kernel <128> : 417 samples, 121 own samples
 * - Compared to ggml_vec_dot_f32 : 250175 samples, 249355 own samples
 * - Compared to ggml_compute_forward_mul_mat : 245494 samples, 3414 own samples
 * The largest matrix multiplies are:
 * mul_mat 29511680 2883584 2682880
 * mul_mat 29511680 2883584 2682880
 * mul_mat 29511680 2883584 2682880
 * mul_mat 29511680 2883584 2682880
 * mul_mat 29511680 2883584 2682880
 * mul_mat 29511680 2883584 2682880
 * mul_mat 88540672 720896 8049152
 * mul_mat 88540672 720896 8049152
 * mul_mat 88540672 720896 8049152
 * mul_mat 88540672 720896 8049152
 * mul_mat 88540672 720896 8049152
 * mul_mat 88540672 720896 8049152
 * mul_mat 24147456 196608 8049152
 * mul_mat 24147456 196608 8049152
 * mul_mat 24147456 196608 8049152
 * mul_mat 24147456 196608 8049152
 * mul_mat 24147456 196608 8049152
 * mul_mat 24147456 196608 8049152
 * mul_mat 56344064 458752 8049152
 * mul_mat 56344064 458752 8049152
 * mul_mat 56344064 458752 8049152
 * mul_mat 56344064 458752 8049152
 * mul_mat 56344064 458752 8049152
 * mul_mat 56344064 458752 8049152
 * mul_mat 88540672 720896 8049152
 * mul_mat 88540672 720896 8049152
 * mul_mat 88540672 720896 8049152
 * mul_mat 88540672 720896 8049152
 * mul_mat 88540672 720896 8049152
 * mul_mat 88540672 720896 8049152
 * mul_mat 56344064 78848 1383448
 */
#ifndef GGML_USE_METAL
template <size_t N_ALPHA>
/* static debug */  void cpu_compute_forward_snake_1d_fused_kernel_impl(ggml_tensor * dst_tensor , const ggml_tensor * a_tensor, const ggml_tensor * alpha_tensor, int ith, int nth) {
    const auto na = N_ALPHA;
    const auto ne{static_cast<size_t>(dst_tensor->ne[1] * na)};
    const std::span dst_ptr{static_cast<float *>(dst_tensor->data), ne};
    const std::span a_ptr{static_cast<const float *>(a_tensor->data), ne};
    const std::span alpha{static_cast<const float *>(alpha_tensor->data), N_ALPHA};
    float alpha_reciprocal[N_ALPHA];
    for (int i = 0; i < N_ALPHA; i++) {
        alpha_reciprocal[i] = 1 / alpha[i];
    }

    for (const auto & [d_row, a_row] : std::views::zip(dst_ptr | std::views::chunk(N_ALPHA), a_ptr | std::views::chunk(N_ALPHA)) | std::views::drop(ith) | std::views::stride(nth)) {
        for (int i0 = 0; i0 < N_ALPHA; i0++) {
            d_row[i0] = sinf(a_row[i0] * alpha[i0]);
        }
        for (int i0 = 0; i0 < N_ALPHA; i0++) {
            d_row[i0] = a_row[i0] + d_row[i0] * d_row[i0] * alpha_reciprocal[i0];
        }
    }
}

/* static debug */ void cpu_compute_forward_snake_1d_fused_kernel(ggml_tensor * dst_tensor , const ggml_tensor * a_tensor, const ggml_tensor * alpha_tensor, int ith, int nth, void *) {
    switch (alpha_tensor->ne[0]) {
        case 256:
            cpu_compute_forward_snake_1d_fused_kernel_impl<256>(dst_tensor, a_tensor, alpha_tensor, ith, nth);
            break;
        case 128:
            cpu_compute_forward_snake_1d_fused_kernel_impl<128>(dst_tensor, a_tensor, alpha_tensor, ith, nth);
            break;
        default:
            TTS_ABORT("alpha_tensor ne assumed to be 256 or 128");
    }
}
static_assert(std::is_same_v<decltype(&cpu_compute_forward_snake_1d_fused_kernel), ggml_custom2_op_t>);
#endif

// Described in https://arxiv.org/abs/2006.08195
// Snake1d is a common tunable activation function used in the DAC model.
ggml_tensor * snake_1d(ggml_context * ctx, ggml_tensor * alpha, ggml_tensor * a) {
    TTS_ASSERT(a->ne[1] == alpha->ne[1] && a->ne[2] == 1 && a->ne[3] == 1 && ggml_is_contiguous(a));
    TTS_ASSERT(alpha->ne[0] == 1 && (alpha->ne[1] == 256 || alpha->ne[1] == 128) && alpha->ne[2] == 1 && alpha->ne[3] == 1 && ggml_is_contiguous(alpha));
    a = ggml_cont(ctx, ggml_transpose(ctx, a));
    alpha = squeeze_2d_1d_e0(ctx, alpha);
#ifdef GGML_USE_METAL
    auto multiplied = ggml_mul(ctx, a, alpha);
    auto sine = ggml_sin(ctx, multiplied);
    auto root = ggml_sqr(ctx, sine);
    auto product = ggml_div(ctx, root, alpha); // change back to reciprocal for caching
    auto result = ggml_add(ctx, a, product);
#else
    auto result = ggml_map_custom2(ctx, a, alpha, cpu_compute_forward_snake_1d_fused_kernel, -1, nullptr);
#endif
    TTS_ASSERT(result->ne[0] == a->ne[0] && result->ne[1] == a->ne[1] && result->ne[2] == a->ne[2] && result->ne[3] == a->ne[3]);
    result = ggml_cont(ctx, ggml_transpose(ctx, result));
    return result;
}
