#include "util.h"

void tts_abort(const char * file, int line, const char * fmt, ...) {
    fflush(stdout);
    fprintf(stderr, "%s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    abort();
}

// TODO: implement this as a ggml op.
// This is a hack to support the reciprocal function needed to implemented the snake operation described below. This is currently very slow and
// should be replaced by a simpler and faster ggml operation.=
struct ggml_tensor * reciprocal(ggml_context * ctx, struct ggml_tensor * a) {
    return ggml_div(ctx, a, ggml_mul(ctx, a, a));
}

// Described in https://arxiv.org/abs/2006.08195
// Snake1d is a common trainable activation function used in the DAC model.
struct ggml_tensor * dac_snake_1d(ggml_context * ctx, struct ggml_tensor * alpha, struct ggml_tensor * a) {
    assert(a->ne[2] == 1 && a->ne[3] == 1);
    return ggml_add(ctx, a, ggml_mul(ctx, ggml_sqr(ctx, ggml_sin(ctx, ggml_mul(ctx, a, alpha))), reciprocal(ctx, alpha)));
}

uint64_t get_cpu_count() {
    uint64_t cpu_count = 0;
    size_t size = sizeof(cpu_count);
    if (sysctlbyname("hw.ncpu", &cpu_count, &size, NULL, 0) < 0) {
        // this functionis only currently used to prepare static cross attention keys and values, and it is fast enough with a single cpu.
        return 1;
    }
    return cpu_count;
}
