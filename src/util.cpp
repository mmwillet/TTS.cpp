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

// Simple helper function for getting layer count from tensor name
std::pair<int, std::string> parse_layer_count(std::string name, int skip) {
    bool found = false;
    bool after_layer = false;
    std::string digit_chars = "";
    std::string after_layer_name = "";
    int count = 0;
    for (char& c : name) {
        if (count < skip) {
            count += 1;
            continue;
        }
        count += 1;
        if (after_layer) {
            after_layer_name += c;
        } else if (std::isdigit(c)) {
            found = true;
            digit_chars += c;
        } else if (!found) {
            
        } else {
            after_layer = true;
            after_layer_name += c;
        }
    }
    if (digit_chars.size() == 0) {
        return std::make_pair(-1, name);
    }
    return std::make_pair(std::stoi(digit_chars), after_layer_name);
}

int search_for_gguf_keys(gguf_context * meta, std::vector<std::string> possible_keys) {
    int gguf_key = -1;
    for (auto key : possible_keys) {
        gguf_key = gguf_find_key(meta, key.c_str());
        if (gguf_key != -1) {
            return gguf_key;
        }
    }
    return gguf_key;
}

void random_gen(int count, float * tgt, float min, float max) {
    static std::default_random_engine e;
    static std::uniform_real_distribution<float> dis(min, max);
    for (int i = 0; i < count; i++) {
        tgt[i] = dis(e);
    }
}

float round_to_float(double v) {
    return roundf(v * powl(10, 6)) / powl(10, 6);
}

// Described in https://arxiv.org/abs/2006.08195
// Snake1d is a common tunable activation function used in the DAC model.
struct ggml_tensor * snake_1d(ggml_context * ctx, struct ggml_tensor * alpha, struct ggml_tensor * a) {
    assert(a->ne[2] == 1 && a->ne[3] == 1);
    return ggml_add(ctx, a, ggml_mul(ctx, ggml_sqr(ctx, ggml_sin(ctx, ggml_mul(ctx, a, alpha))), ggml_reciprocal(ctx, alpha)));
}

bool has_suffix(std::string value, std::string suffix) {
    return value.size() >= suffix.size() && value.compare(value.size()-suffix.size(), suffix.size(), suffix) == 0;
}

bool has_prefix(std::string value, std::string prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

struct ggml_tensor * stft(ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * window, size_t n_fft, size_t hop, bool abs_and_angle, bool one_sided) {
    if (window->ne[0] != n_fft) {
        TTS_ABORT("For #stft the window_size, %d, must be either equal to n_fft, %d, or, when one sided, n_fft / 2 + 1, %d.\n", a->ne[0], n_fft, n_fft/2+1);
    }
    struct ggml_tensor * cur = ggml_stft(ctx, a, window, n_fft, hop, abs_and_angle);
    if (one_sided) {
        cur = ggml_cont(ctx, ggml_view_4d(ctx, cur, ((int64_t) n_fft / 2) + 1, cur->ne[1], cur->ne[2], cur->ne[3], cur->nb[1], cur->nb[2], cur->nb[3], 0));
    }

    return cur;
}

struct ggml_tensor * istft(ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * window_squared_sum, struct ggml_tensor * window, size_t n_fft, size_t hop, bool abs_and_angle, bool one_sided) {
    if ((!one_sided && a->ne[0] != n_fft) || (one_sided && a->ne[0] != n_fft / 2 + 1)) {
        TTS_ABORT("For #istft the window_size, %d, must be either equal to n_fft, %d, or, when one sided, n_fft / 2 + 1, %d.\n", a->ne[0], n_fft, n_fft/2+1);
    }
    struct ggml_tensor * cur = ggml_istft(ctx, a, window, n_fft, hop, abs_and_angle);
    cur = ggml_div(ctx, cur, window_squared_sum);
    return cur;
}

void hann_window(size_t n_fft, float * tgt) {
    for (int i = 0; i < n_fft; i++) {
        float v = pow(sin(M_PI * (double)i / (double) n_fft), 2.0);
        tgt[i] = v;
    }
}

// This is a custom map op for computing noise and relevant voiced sections.
void uv_noise_compute(struct ggml_tensor * dst, const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata) {
    float voice_threshold = ((float *) userdata)[0];
    float noise_std = ((float *) userdata)[1];
    float sin_amp = ((float *) userdata)[2];
    float sin_amp_div = ((float *) userdata)[3];
    float * rand_init = ((float *) userdata) + 4;

    const int rpt = (b->ne[0]) / nth + 1;
    const int start = ith * rpt;
    const int end = MIN((ith + 1) * rpt, b->ne[0]);

    float * uv_dst = (float *) dst->data;
    float * noise_dst = (float *)((char*)dst->data + dst->nb[2]);
    float * tgt = (float *) b->data;

    for(int bt = 0; bt < b->ne[2]; bt++) {
        for(int r = start; r < end; r++) {
            if (tgt[r] > voice_threshold) {
                for (int h = 0; h < a->ne[1]; h++) {
                    int index = h*dst->ne[0]+r;
                    uv_dst[index] = sin_amp;
                    noise_dst[index] = noise_std * rand_init[index];
                }
            } else {
                for (int h = 0; h < a->ne[1]; h++) {
                    int index = h*dst->ne[0]+r;
                    uv_dst[index] = 0.0f;
                    noise_dst[index] = sin_amp_div * rand_init[index];
                }
            }
        }
    }
}

// currently this assumes a center view in which the output vector is reflectively padded by n_fft / 2 on each side.
void compute_window_squared_sum(size_t n_fft, size_t hop, size_t n_frames, float * tgt, float * window) {
    size_t out_size = n_frames * hop;
    size_t half = n_fft / 2;
    size_t cutoff = out_size - n_fft;
    for (int i = 0; i < n_frames; i++) {
        for (int ii = 0; ii < n_fft; ii++) {
            int index = ii + i*hop - half;
            if (index < 0 || index >= cutoff) {
                continue;
            }
            tgt[index] += powf(window[ii], 2);
        }
    }
}

std::string replace_any(std::string target, std::string to_replace, std::string replacement) {
    for (int i = 0; i < to_replace.size(); i++) {
        size_t position = target.find(to_replace[i]);
        while (position != std::string::npos) {
            target.replace(position, 1, replacement);
            position = target.find(to_replace[i]);
        }
    }
    return target;
}

struct model_tensor_meta compute_tensor_meta(std::string name_prefix, ggml_context * weight_ctx, std::function<void(ggml_tensor*)>* callback) {
    model_tensor_meta meta;
    for (ggml_tensor * cur = ggml_get_first_tensor(weight_ctx); cur; cur = ggml_get_next_tensor(weight_ctx, cur)) {
        if (callback) {
            (*callback)(cur);
        }
        std::string::size_type pos = std::string(cur->name).find(".", 0);
        std::string top_level(std::string(cur->name).substr(0, pos));
        if (top_level == name_prefix) {
            meta.n_tensors += 1;
            meta.n_bytes += ggml_nbytes_pad(cur);
        }
    }
    return meta;
}
