#include "ggml-tts-ext.h"
#include <cmath>
#include <complex>
#include <cstring>
#include <vector>

// Helper functions for FFT computation
static void fft(std::vector<std::complex<float>> &x) {
  const size_t N = x.size();
  if (N <= 1)
    return;

  // Divide
  std::vector<std::complex<float>> even(N / 2), odd(N / 2);
  for (size_t i = 0; i < N / 2; i++) {
    even[i] = x[i * 2];
    odd[i] = x[i * 2 + 1];
  }

  // Conquer
  fft(even);
  fft(odd);

  // Combine
  for (size_t i = 0; i < N / 2; i++) {
    std::complex<float> t =
        std::polar(1.0f, -2.0f * (float)M_PI * (float)i / (float)N) * odd[i];
    x[i] = even[i] + t;
    x[i + N / 2] = even[i] - t;
  }
}

static void ifft(std::vector<std::complex<float>> &x) {
  // Conjugate the complex numbers
  for (auto &val : x) {
    val = std::conj(val);
  }

  // Forward FFT
  fft(x);

  // Conjugate the complex numbers again and scale
  for (auto &val : x) {
    val = std::conj(val) / static_cast<float>(x.size());
  }
}

// Custom compute functions for map operations
static void ggml_compute_mod_f32(struct ggml_tensor *dst,
                                 const struct ggml_tensor *src, int ith,
                                 int nth, void *userdata) {
  const float mod_val = *(float *)userdata;
  const int n = ggml_nrows(src);
  const int nc = src->ne[0];

  for (int i = ith; i < n; i += nth) {
    const float *src_row = (float *)((char *)src->data + i * src->nb[1]);
    float *dst_row = (float *)((char *)dst->data + i * dst->nb[1]);

    for (int j = 0; j < nc; j++) {
      dst_row[j] = fmodf(src_row[j], mod_val);
    }
  }
}

static void ggml_compute_cumsum(struct ggml_tensor *dst,
                                const struct ggml_tensor *src, int ith, int nth,
                                void *userdata) {
  const int n = ggml_nrows(src);
  const int nc = src->ne[0];

  for (int i = ith; i < n; i += nth) {
    const float *src_row = (float *)((char *)src->data + i * src->nb[1]);
    float *dst_row = (float *)((char *)dst->data + i * dst->nb[1]);

    float sum = 0.0f;
    for (int j = 0; j < nc; j++) {
      sum += src_row[j];
      dst_row[j] = sum;
    }
  }
}

static void ggml_compute_upscale_linear(struct ggml_tensor *dst,
                                        const struct ggml_tensor *src,
                                        const struct ggml_tensor *param,
                                        int ith, int nth, void *userdata) {
  const int scale_factor = *(int *)param->data;
  const int n = ggml_nrows(src);
  const int nc = src->ne[0];

  for (int i = ith; i < n; i += nth) {
    const float *src_row = (float *)((char *)src->data + i * src->nb[1]);
    float *dst_row = (float *)((char *)dst->data + i * dst->nb[1]);

    for (int j = 0; j < nc; j++) {
      for (int k = 0; k < scale_factor; k++) {
        dst_row[j * scale_factor + k] = src_row[j];
      }
    }
  }
}

static void ggml_compute_round(struct ggml_tensor *dst,
                               const struct ggml_tensor *src, int ith, int nth,
                               void *userdata) {
  const int n = ggml_nrows(src);
  const int nc = src->ne[0];

  for (int i = ith; i < n; i += nth) {
    const float *src_row = (float *)((char *)src->data + i * src->nb[1]);
    float *dst_row = (float *)((char *)dst->data + i * dst->nb[1]);

    for (int j = 0; j < nc; j++) {
      dst_row[j] = roundf(src_row[j]);
    }
  }
}

// STFT computation parameters
struct stft_params {
  int filter_length;
  int hop_length;
  bool compute_abs_and_angle;
};

static void ggml_compute_stft(struct ggml_tensor *dst,
                              const struct ggml_tensor *src,
                              const struct ggml_tensor *window, int ith,
                              int nth, void *userdata) {
  const stft_params *params = (const stft_params *)userdata;
  const int filter_length = params->filter_length;
  const int hop_length = params->hop_length;
  const bool compute_abs_and_angle = params->compute_abs_and_angle;

  const int input_length = src->ne[0];
  const int n_frames = (input_length - filter_length) / hop_length + 1;
  const int freq_bins = filter_length;

  const float *input = (const float *)src->data;
  const float *win = (const float *)window->data;
  float *output = (float *)dst->data;

  // Process each frame
  for (int frame = ith; frame < n_frames; frame += nth) {
    const int start_idx = frame * hop_length;

    // Prepare windowed frame for FFT
    std::vector<std::complex<float>> frame_data(filter_length);
    for (int i = 0; i < filter_length; i++) {
      if (start_idx + i < input_length) {
        frame_data[i] =
            std::complex<float>(input[start_idx + i] * win[i], 0.0f);
      } else {
        frame_data[i] = std::complex<float>(0.0f, 0.0f);
      }
    }

    // Compute FFT
    fft(frame_data);

    // Store results
    for (int freq = 0; freq < freq_bins; freq++) {
      if (compute_abs_and_angle) {
        // Store magnitude and phase
        float magnitude = std::abs(frame_data[freq]);
        float phase = std::arg(frame_data[freq]);
        output[freq * n_frames * 2 + frame * 2 + 0] = magnitude;
        output[freq * n_frames * 2 + frame * 2 + 1] = phase;
      } else {
        // Store real and imaginary parts
        output[freq * n_frames * 2 + frame * 2 + 0] = frame_data[freq].real();
        output[freq * n_frames * 2 + frame * 2 + 1] = frame_data[freq].imag();
      }
    }
  }
}

// ISTFT computation parameters
struct istft_params {
  int filter_length;
  int hop_length;
  bool from_abs_and_angle;
};

static void ggml_compute_istft(struct ggml_tensor *dst,
                               const struct ggml_tensor *src,
                               const struct ggml_tensor *window, int ith,
                               int nth, void *userdata) {
  const istft_params *params = (const istft_params *)userdata;
  const int filter_length = params->filter_length;
  const int hop_length = params->hop_length;
  const bool from_abs_and_angle = params->from_abs_and_angle;

  const int n_frames = src->ne[1];
  const int freq_bins = src->ne[0];
  const int output_length = (n_frames - 1) * hop_length + filter_length;

  const float *input = (const float *)src->data;
  const float *win = (const float *)window->data;
  float *output = (float *)dst->data;

  // Initialize output to zero
  if (ith == 0) {
    memset(output, 0, output_length * sizeof(float));
  }

  // Process each frame
  for (int frame = ith; frame < n_frames; frame += nth) {
    const int start_idx = frame * hop_length;

    // Prepare frequency domain data for IFFT
    std::vector<std::complex<float>> frame_data(filter_length);
    for (int freq = 0; freq < freq_bins && freq < filter_length; freq++) {
      if (from_abs_and_angle) {
        // Convert from magnitude and phase
        float magnitude = input[freq * n_frames * 2 + frame * 2 + 0];
        float phase = input[freq * n_frames * 2 + frame * 2 + 1];
        frame_data[freq] = std::polar(magnitude, phase);
      } else {
        // Use real and imaginary parts
        float real = input[freq * n_frames * 2 + frame * 2 + 0];
        float imag = input[freq * n_frames * 2 + frame * 2 + 1];
        frame_data[freq] = std::complex<float>(real, imag);
      }
    }

    // Fill remaining frequencies with conjugate symmetry for real output
    for (int freq = freq_bins; freq < filter_length; freq++) {
      int mirror_freq = filter_length - freq;
      if (mirror_freq < freq_bins) {
        frame_data[freq] = std::conj(frame_data[mirror_freq]);
      } else {
        frame_data[freq] = std::complex<float>(0.0f, 0.0f);
      }
    }

    // Compute IFFT
    ifft(frame_data);

    // Apply window and overlap-add
    for (int i = 0; i < filter_length; i++) {
      if (start_idx + i < output_length) {
        output[start_idx + i] += frame_data[i].real() * win[i];
      }
    }
  }
}

// Modulo operation: a % b
struct ggml_tensor *ggml_mod(struct ggml_context *ctx, struct ggml_tensor *a,
                             float b) {
  float *mod_val = (float *)malloc(sizeof(float));
  *mod_val = b;

  return ggml_map_custom1(ctx, a, ggml_compute_mod_f32, 1, mod_val);
}

// Cumulative sum along the first dimension
struct ggml_tensor *ggml_cumsum(struct ggml_context *ctx,
                                struct ggml_tensor *a) {
  return ggml_map_custom1(ctx, a, ggml_compute_cumsum, 1, nullptr);
}

// Linear upscaling (repeat elements)
struct ggml_tensor *ggml_upscale_linear(struct ggml_context *ctx,
                                        struct ggml_tensor *a,
                                        int scale_factor) {
  // Store scale factor in a parameter tensor
  struct ggml_tensor *scale_param = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
  *(int *)scale_param->data = scale_factor;

  return ggml_map_custom2(ctx, a, scale_param, ggml_compute_upscale_linear, 1,
                          nullptr);
}

// Round to nearest integer
struct ggml_tensor *ggml_round(struct ggml_context *ctx,
                               struct ggml_tensor *a) {
  return ggml_map_custom1(ctx, a, ggml_compute_round, 1, nullptr);
}

// Helper function to calculate number of frames
static int64_t calculate_number_of_frames(int64_t length, size_t hop_length) {
  return (int64_t)(length / (int64_t)hop_length) + 1;
}

// Helper function to calculate original length
static int64_t calculate_original_length(int64_t frames, size_t hop_length) {
  return (frames - 1) * (int64_t)hop_length;
}

// STFT implementation - adapted from ggml fork
struct ggml_tensor *stft(struct ggml_context *ctx, struct ggml_tensor *a,
                         struct ggml_tensor *window, size_t n_fft, size_t hop,
                         bool abs_and_angle, bool one_sided) {
  // Calculate output dimensions
  int64_t n_frames = calculate_number_of_frames(a->ne[0], hop);
  int64_t freq_bins = one_sided ? (n_fft / 2 + 1) : n_fft;

  // Output tensor: [freq_bins, n_frames, 2] where last dim is real/imag or
  // mag/phase
  int64_t ne[4] = {freq_bins, n_frames, 2, 1};
  struct ggml_tensor *result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

  // Create parameters
  stft_params *params = (stft_params *)malloc(sizeof(stft_params));
  params->filter_length = n_fft;
  params->hop_length = hop;
  params->compute_abs_and_angle = abs_and_angle;

  return ggml_map_custom2(ctx, a, window, ggml_compute_stft, 1, params);
}

// ISTFT implementation - adapted from ggml fork
struct ggml_tensor *istft(struct ggml_context *ctx, struct ggml_tensor *a,
                          struct ggml_tensor *window_squared_sum,
                          struct ggml_tensor *window, size_t n_fft, size_t hop,
                          bool abs_and_angle, bool one_sided) {
  // Calculate output dimensions
  int64_t n_frames = a->ne[1];
  int64_t output_length = calculate_original_length(n_frames, hop) + n_fft;

  int64_t ne[4] = {output_length, 1, 1, 1};
  struct ggml_tensor *result = ggml_new_tensor(ctx, GGML_TYPE_F32, 1, ne);

  // Create parameters
  istft_params *params = (istft_params *)malloc(sizeof(istft_params));
  params->filter_length = n_fft;
  params->hop_length = hop;
  params->from_abs_and_angle = abs_and_angle;

  return ggml_map_custom2(ctx, a, window, ggml_compute_istft, 1, params);
}