#include "util.h"

#include <algorithm>
#include <cstdio>
#include <stdarg.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#elif __linux__
#include <unistd.h>
#else
// windows stuff
#endif
#include "tts_ggml_iterator.h"

// Simple helper function for getting layer count from tensor name
pair<uint32_t, sv> parse_layer_count(sv name) {
    const size_t start{name.find_first_of("0123456789")};
    if (!~start) {
        return {~0, {}};
    }
    name.remove_prefix(start);
    const size_t end{name.find_first_not_of("0123456789")};
    if (!~end) {
        return {sv_int(name), {}};
    }
    return {sv_int(name.substr(0, end)), name.substr(end)};
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

void hann_window(size_t n_fft, std::vector<float> & tgt) {
    for (int i = 0; i < n_fft; i++) {
        float v = pow(sin(M_PI * (double)i / (double) n_fft), 2.0);
        tgt.push_back(v);
    }
}

// This is a custom map op for computing noise and relevant voiced sections.
void uv_noise_compute(struct ggml_tensor * dst, const struct ggml_tensor * a, const struct ggml_tensor * b, const struct ggml_tensor * c, int ith, int nth, void * userdata) {
    float voice_threshold = ((float *) c->data)[0];
    float noise_std = ((float *) c->data)[1];
    float sin_amp = ((float *) c->data)[2];
    float sin_amp_div = ((float *) c->data)[3];
    float * rand_init = ((float *) c->data) + 4;

    const int rpt = (b->ne[0] + nth - 1)/nth;
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

// This is a custom map op for applying cfg scale. It is used at the terminus of logit generation in Dia.
void cfg_scale(struct ggml_tensor * dst, const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata) {
    const float scale = ((float *) userdata)[0];
    const float max_output = ((float*) userdata)[1];
    const int rpt = (b->ne[0] + nth - 1)/nth;
    const int start = ith * rpt;
    const int end = MIN((ith + 1) * rpt, b->ne[0]);

    float * output = (float *) dst->data;
    float * cond = (float *) a->data;
    float * uncond = (float *) b->data;

    for(int bt = 0; bt < b->ne[2]; bt++) {
        for (int h = 0; h < b->ne[1]; h++) {
            int i = (h * b->ne[0]) + (bt * b->ne[0] * b->ne[1]);
            for(int r = start; r < end; r++) {
                // only let the output heads yield tokens up to EOS
                if (r > max_output) {
                    output[i+r] = -INFINITY;
                }
                const float cr = cond[i+r];
                const float ur = uncond[i+r];
                output[i+r] = cr + scale * (cr - ur);
            }
        }
    }
}

// currently this assumes a center view in which the output vector is reflectively padded by n_fft / 2 on each side.
void compute_window_squared_sum(size_t n_fft, size_t hop, size_t n_frames, float * tgt, float * window) {
    size_t cutoff = n_frames * hop;
    size_t half = n_fft / 2;
    std::memset(tgt, 0, cutoff*sizeof(float));
    // istft applies half / hop steps before the beginning of the sequence. We need to account for these accumulated windows.
    for (int i = 0; i < n_frames + (half / hop); i++) {
        for (int ii = 0; ii < n_fft; ii++) {
            int index = ii + i*hop - half;
            if (index < 0 || index >= cutoff) {
                continue;
            }
            tgt[index] += powf(window[ii], 2);
        }
    }
}

string strip(const string & target, const string & vals) {
	string result{target};
    result.erase(result.begin(), std::find_if(result.begin(), result.end(), [&vals](unsigned char ch) {
        return vals.find(ch) == std::string::npos;
    }));
    result.erase(std::find_if(result.rbegin(), result.rend(), [&vals](unsigned char ch) {
        return vals.find(ch) == std::string::npos;
    }).base(), result.end());
    return result;
}

model_tensor_meta::model_tensor_meta(ggml_context * weight_ctx, sv name_prefix,
                                     uint8_t size_multiplier_percent,
                                     uint32_t dedicated_add_on_size) : n_tensors{}, buf_size{} {
    for (const ggml_tensor & tensor: ggml_tensor_iterator{weight_ctx}) {
        if (const sv name{tensor.name}; name.starts_with(name_prefix) && name[name_prefix.size()] == '.') {
            n_tensors += 1;
            buf_size += ggml_nbytes_pad(&tensor);
        }
    }
    buf_size += dedicated_add_on_size;
    mem_size = ggml_tensor_overhead() * n_tensors * size_multiplier_percent / 100;
}
