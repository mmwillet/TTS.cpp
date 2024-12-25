#include "util.h"

#include <cstdio>
#include <stdarg.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#elif __linux__
#include <unistd.h>
#else
// windows stuff
#endif

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
// Snake1d is a common tunable activation function used in the DAC model.
struct ggml_tensor * dac_snake_1d(ggml_context * ctx, struct ggml_tensor * alpha, struct ggml_tensor * a) {
    assert(a->ne[2] == 1 && a->ne[3] == 1);
    return ggml_add(ctx, a, ggml_mul(ctx, ggml_sqr(ctx, ggml_sin(ctx, ggml_mul(ctx, a, alpha))), ggml_reciprocal(ctx, alpha)));
}

uint64_t get_cpu_count() {
    uint64_t cpu_count = 0;
    size_t size = sizeof(cpu_count);
#ifdef __APPLE__
    if (sysctlbyname("hw.ncpu", &cpu_count, &size, NULL, 0) < 0) {
        // this functionis only currently used to prepare static cross attention keys and values, and it is fast enough with a single cpu.
        return 1;
    }
#elif __linux__
    cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpu_count == -1){
        return 1;
    }
#else
    // windows stuff
#endif
    return cpu_count;
}

bool has_suffix(std::string value, std::string suffix) {
    return value.size() >= suffix.size() && value.compare(value.size()-suffix.size(), suffix.size(), suffix) == 0;
}

bool has_prefix(std::string value, std::string prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

struct model_tensor_meta compute_tensor_meta(std::string name_prefix, ggml_context * weight_ctx) {
    model_tensor_meta meta;
    for (ggml_tensor * cur = ggml_get_first_tensor(weight_ctx); cur; cur = ggml_get_next_tensor(weight_ctx, cur)) {
        std::string::size_type pos = std::string(cur->name).find(".", 0);
        std::string top_level(std::string(cur->name).substr(0, pos));
        if (top_level == name_prefix) {
            meta.n_tensors += 1;
            meta.n_bytes += ggml_nbytes_pad(cur);
        }
    }
    return meta;
}
