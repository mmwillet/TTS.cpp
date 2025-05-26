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

void tts_abort(const char *file, int line, const char *fmt, ...) {
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
  for (char &c : name) {
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

int search_for_gguf_keys(gguf_context *meta,
                         std::vector<std::string> possible_keys) {
  int gguf_key = -1;
  for (auto key : possible_keys) {
    gguf_key = gguf_find_key(meta, key.c_str());
    if (gguf_key != -1) {
      return gguf_key;
    }
  }
  return gguf_key;
}

void random_gen(int count, float *tgt, float min, float max) {
  static std::default_random_engine e;
  static std::uniform_real_distribution<float> dis(min, max);
  for (int i = 0; i < count; i++) {
    tgt[i] = dis(e);
  }
}

float round_to_float(double v) { return roundf(v * powl(10, 6)) / powl(10, 6); }

// Helper function for alpha broadcasting in snake_1d (same as bias
// broadcasting)
static struct ggml_tensor *prepare_alpha_for_mul(struct ggml_context *ctx,
                                                 struct ggml_tensor *alpha,
                                                 struct ggml_tensor *target) {
  // For Snake activation: alpha needs to broadcast across the time dimension
  // (dim 0) but match the channel dimension (dim 1) Original alpha shape: [1,
  // N, 1, 1] or [N, 1, 1, 1] Target shape: [time_steps, N, 1, 1] We need alpha
  // to be [1, N, 1, 1] so it can broadcast to [time_steps, N, 1, 1]

  // Check if alpha is already in the right shape for broadcasting
  if (alpha->ne[0] == 1 && alpha->ne[1] == target->ne[1] && alpha->ne[2] == 1 &&
      alpha->ne[3] == 1) {
    // Already in correct shape [1, N, 1, 1]
    return alpha;
  }

  // If alpha is [N, 1, 1, 1], reshape to [1, N, 1, 1]
  if (alpha->ne[0] == target->ne[1] && alpha->ne[1] == 1 && alpha->ne[2] == 1 &&
      alpha->ne[3] == 1) {
    return ggml_reshape_4d(ctx, alpha, 1, alpha->ne[0], 1, 1);
  }

  // Default: try to reshape to [1, channels, 1, 1]
  return ggml_reshape_4d(ctx, alpha, 1, target->ne[1], 1, 1);
}

// Described in https://arxiv.org/abs/2006.08195
// Snake1d is a common tunable activation function used in the DAC model.
struct ggml_tensor *snake_1d(ggml_context *ctx, struct ggml_tensor *alpha,
                             struct ggml_tensor *x) {
  // Simplified Snake activation that avoids division issues
  // Based on the observation that Snake(x) = x + α * sin²(x/α)
  // We can approximate this for small α as: x + α * sin²(x) * scale_factor
  // This avoids the problematic x/α division

  // Prepare alpha for broadcasting to match x's shape
  struct ggml_tensor *alpha_broadcast = prepare_alpha_for_mul(ctx, alpha, x);

  // Compute sin(x) - this is stable
  struct ggml_tensor *sin_x = ggml_sin(ctx, x);

  // Compute sin²(x) = sin(x) * sin(x)
  struct ggml_tensor *sin_squared = ggml_mul(ctx, sin_x, sin_x);

  // Scale by alpha: sin²(x) * α (swapped order for correct broadcasting)
  struct ggml_tensor *alpha_sin_squared =
      ggml_mul(ctx, sin_squared, alpha_broadcast);

  // Final result: x + α * sin²(x)
  return ggml_add(ctx, x, alpha_sin_squared);
}

bool has_suffix(std::string value, std::string suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
             0;
}

bool has_prefix(std::string value, std::string prefix) {
  return value.size() >= prefix.size() &&
         value.compare(0, prefix.size(), prefix) == 0;
}

// STFT/ISTFT functions removed - they were using custom GGML operations
// that don't exist in standard llama.cpp/ggml. If needed, they should be
// implemented using standard GGML operations or as custom map operations.

void hann_window(size_t n_fft, std::vector<float> &tgt) {
  for (int i = 0; i < n_fft; i++) {
    float v = pow(sin(M_PI * (double)i / (double)n_fft), 2.0);
    tgt.push_back(v);
  }
}

// This is a custom map op for computing noise and relevant voiced sections.
void uv_noise_compute(struct ggml_tensor *dst, const struct ggml_tensor *a,
                      const struct ggml_tensor *b, const struct ggml_tensor *c,
                      int ith, int nth, void *userdata) {
  float voice_threshold = ((float *)c->data)[0];
  float noise_std = ((float *)c->data)[1];
  float sin_amp = ((float *)c->data)[2];
  float sin_amp_div = ((float *)c->data)[3];
  float *rand_init = ((float *)c->data) + 4;

  const int rpt = (b->ne[0] + nth - 1) / nth;
  const int start = ith * rpt;
  const int end = MIN((ith + 1) * rpt, b->ne[0]);

  float *uv_dst = (float *)dst->data;
  float *noise_dst = (float *)((char *)dst->data + dst->nb[2]);
  float *tgt = (float *)b->data;

  for (int bt = 0; bt < b->ne[2]; bt++) {
    for (int r = start; r < end; r++) {
      if (tgt[r] > voice_threshold) {
        for (int h = 0; h < a->ne[1]; h++) {
          int index = h * dst->ne[0] + r;
          uv_dst[index] = sin_amp;
          noise_dst[index] = noise_std * rand_init[index];
        }
      } else {
        for (int h = 0; h < a->ne[1]; h++) {
          int index = h * dst->ne[0] + r;
          uv_dst[index] = 0.0f;
          noise_dst[index] = sin_amp_div * rand_init[index];
        }
      }
    }
  }
}

// This is a custom map op for applying cfg scale. It is used at the terminus of
// logit generation in Dia.
void cfg_scale(struct ggml_tensor *dst, const struct ggml_tensor *a,
               const struct ggml_tensor *b, int ith, int nth, void *userdata) {
  const float scale = ((float *)userdata)[0];
  const float max_output = ((float *)userdata)[1];
  const int rpt = (b->ne[0] + nth - 1) / nth;
  const int start = ith * rpt;
  const int end = MIN((ith + 1) * rpt, b->ne[0]);

  float *output = (float *)dst->data;
  float *cond = (float *)a->data;
  float *uncond = (float *)b->data;

  for (int bt = 0; bt < b->ne[2]; bt++) {
    for (int h = 0; h < b->ne[1]; h++) {
      int i = (h * b->ne[0]) + (bt * b->ne[0] * b->ne[1]);
      for (int r = start; r < end; r++) {
        // only let the output heads yield tokens up to EOS
        if (r > max_output) {
          output[i + r] = -INFINITY;
        }
        const float cr = cond[i + r];
        const float ur = uncond[i + r];
        output[i + r] = cr + scale * (cr - ur);
      }
    }
  }
}

// currently this assumes a center view in which the output vector is
// reflectively padded by n_fft / 2 on each side.
void compute_window_squared_sum(size_t n_fft, size_t hop, size_t n_frames,
                                float *tgt, float *window) {
  size_t cutoff = n_frames * hop;
  size_t half = n_fft / 2;
  std::memset(tgt, 0, cutoff * sizeof(float));
  // istft applies half / hop steps before the beginning of the sequence. We
  // need to account for these accumulated windows.
  for (int i = 0; i < n_frames + (half / hop); i++) {
    for (int ii = 0; ii < n_fft; ii++) {
      int index = ii + i * hop - half;
      if (index < 0 || index >= cutoff) {
        continue;
      }
      tgt[index] += powf(window[ii], 2);
    }
  }
}

std::vector<std::string> split(std::string target, std::string split_on,
                               bool include_split_characters) {
  std::vector<std::string> output;
  size_t last = 0;

  for (int i = 0; i < target.size(); i++) {
    if (i > last && split_on.find(target[i]) != std::string::npos) {
      std::string part(target.substr(last, i - last));
      output.push_back(part);
      if (include_split_characters) {
        output.push_back(target.substr(i, 1));
      }
      last = i + 1;
    }
  }
  if (last < target.size()) {
    std::string part(target.substr(last));
    output.push_back(part);
  }

  return output;
}

std::vector<std::string> split(std::string target, const char split_on,
                               bool include_split_characters) {
  std::vector<std::string> output;
  size_t last = 0;

  for (int i = 0; i < target.size(); i++) {
    if (i > last && split_on == target[i]) {
      std::string part(target.substr(last, i - last));
      output.push_back(part);
      if (include_split_characters) {
        output.push_back(target.substr(i, 1));
      }
      last = i + 1;
    }
  }
  if (last < target.size()) {
    std::string part(target.substr(last));
    output.push_back(part);
  }

  return output;
}

std::string strip(std::string target, std::string vals) {
  target.erase(target.begin(), std::find_if(target.begin(), target.end(),
                                            [&vals](unsigned char ch) {
                                              return vals.find(ch) ==
                                                     std::string::npos;
                                            }));
  target.erase(std::find_if(target.rbegin(), target.rend(),
                            [&vals](unsigned char ch) {
                              return vals.find(ch) == std::string::npos;
                            })
                   .base(),
               target.end());
  return target;
}

std::string replace_any(std::string target, std::string to_replace,
                        std::string replacement) {
  for (int i = 0; i < to_replace.size(); i++) {
    size_t position = target.find(to_replace[i]);
    while (position != std::string::npos) {
      target.replace(position, 1, replacement);
      position = target.find(to_replace[i]);
    }
  }
  return target;
}

struct model_tensor_meta
compute_tensor_meta(std::string name_prefix, ggml_context *weight_ctx,
                    std::function<void(ggml_tensor *)> *callback) {
  model_tensor_meta meta;
  for (ggml_tensor *cur = ggml_get_first_tensor(weight_ctx); cur;
       cur = ggml_get_next_tensor(weight_ctx, cur)) {
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
