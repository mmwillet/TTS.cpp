#include "util.h"

ggml_tensor * reciprocal(ggml_context * ctx, ggml_tensor * x) {
    TTS_ASSERT(x->ne[0] == 1);
    static constexpr float one = 1.0f;
    ggml_tensor * numerator = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, x->ne[1]);
    // stride trick so that the scalar numerator can be divided by x.
    numerator->nb[1] = 0;
    numerator->data = const_cast<float *>(&one);
    return ggml_div(ctx, numerator, x);
}

// Described in https://arxiv.org/abs/2006.08195
// Snake1d is a common tunable activation function used in the DAC model.
ggml_tensor * snake_1d(ggml_context * ctx, ggml_tensor * alpha, ggml_tensor * a) {
    assert(a->ne[2] == 1 && a->ne[3] == 1);
    return ggml_add(ctx, a, ggml_mul(ctx, ggml_sqr(ctx, ggml_sin(ctx, ggml_mul(ctx, a, alpha))), reciprocal(ctx, alpha)));
}
