#include "ggml-tts-ext.h"
#include <algorithm>
#include <cmath>
#include <cstring>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CACHE_LINE_SIZE_F32 (64 / sizeof(float))

// Custom operation implementations using ggml_map_custom

// Modulo operation
static void ggml_compute_mod_f32(struct ggml_tensor *dst,
                                 const struct ggml_tensor *src0, int ith,
                                 int nth, void *userdata) {
  float mod_val = *(float *)userdata;
  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  for (int i = ith; i < n; i += nth) {
    const float *src_row = (float *)((char *)src0->data + i * src0->nb[1]);
    float *dst_row = (float *)((char *)dst->data + i * dst->nb[1]);

    for (int j = 0; j < nc; j++) {
      dst_row[j] = fmodf(src_row[j], mod_val);
    }
  }
}

struct ggml_tensor *ggml_mod(struct ggml_context *ctx, struct ggml_tensor *a,
                             float b) {
  float *mod_val = (float *)malloc(sizeof(float));
  *mod_val = b;

  return ggml_map_custom1(ctx, a, ggml_compute_mod_f32, 1, mod_val);
}

// Cumulative sum operation
static void ggml_compute_cumsum(struct ggml_tensor *dst,
                                const struct ggml_tensor *src0, int ith,
                                int nth, void *userdata) {
  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  for (int i = ith; i < n; i += nth) {
    const float *src_row = (float *)((char *)src0->data + i * src0->nb[1]);
    float *dst_row = (float *)((char *)dst->data + i * dst->nb[1]);

    float cumsum = 0.0f;
    for (int j = 0; j < nc; j++) {
      cumsum += src_row[j];
      dst_row[j] = cumsum;
    }
  }
}

struct ggml_tensor *ggml_cumsum(struct ggml_context *ctx,
                                struct ggml_tensor *a) {
  return ggml_map_custom1(ctx, a, ggml_compute_cumsum, 1, NULL);
}

// Linear upscaling operation
static void ggml_compute_upscale_linear(struct ggml_tensor *dst,
                                        const struct ggml_tensor *src0, int ith,
                                        int nth, void *userdata) {
  int scale_factor = *(int *)userdata;
  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  for (int i = ith; i < n; i += nth) {
    const float *src_row = (float *)((char *)src0->data + i * src0->nb[1]);
    float *dst_row = (float *)((char *)dst->data + i * dst->nb[1]);

    for (int j = 0; j < nc; j++) {
      for (int k = 0; k < scale_factor; k++) {
        dst_row[j * scale_factor + k] = src_row[j];
      }
    }
  }
}

struct ggml_tensor *ggml_upscale_linear(struct ggml_context *ctx,
                                        struct ggml_tensor *a,
                                        int scale_factor) {
  int *scale_val = (int *)malloc(sizeof(int));
  *scale_val = scale_factor;

  // Create output tensor with scaled dimensions
  int64_t ne[4] = {a->ne[0] * scale_factor, a->ne[1], a->ne[2], a->ne[3]};
  struct ggml_tensor *result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

  return ggml_map_custom1(ctx, a, ggml_compute_upscale_linear, 1, scale_val);
}

// Round operation
static void ggml_compute_round_f32(struct ggml_tensor *dst,
                                   const struct ggml_tensor *src0, int ith,
                                   int nth, void *userdata) {
  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  for (int i = ith; i < n; i += nth) {
    const float *src_row = (float *)((char *)src0->data + i * src0->nb[1]);
    float *dst_row = (float *)((char *)dst->data + i * dst->nb[1]);

    for (int j = 0; j < nc; j++) {
      dst_row[j] = roundf(src_row[j]);
    }
  }
}

struct ggml_tensor *ggml_round(struct ggml_context *ctx,
                               struct ggml_tensor *a) {
  return ggml_map_custom1(ctx, a, ggml_compute_round_f32, 1, NULL);
}

// ============================================================================
// STFT/ISTFT Implementation - Exact copy from original ggml fork
// ============================================================================

// Helper functions for calculating STFT dimensions
static int calculate_number_of_frames(int signal_length, int hop_length) {
  return (signal_length + hop_length - 1) / hop_length;
}

static int calculate_original_length(int n_frames, int hop_length) {
  return (n_frames - 1) * hop_length;
}

// Simple O(N*N) DFT implementation for odd-sized FFTs
static void simple_dft(float *mdst, float *phdst, float *buffer, size_t n_fft,
                       size_t step) {
  float base_k = M_PI * -2.0f / (float)n_fft;
  for (int i = 0; i < n_fft; i++) {
    float tm = 0.0f;
    float tph = 0.0f;
    for (int ii = 0; ii < n_fft; ii++) {
      float k = base_k * (float)ii * (float)i;
      float m = cosf(k);
      float expk = sinf(k);
      tm += mdst[ii * step] * m - phdst[ii * step] * expk;
      tph += mdst[ii * step] * expk + phdst[ii * step] * m;
    }
    buffer[i * 2] = tm;
    buffer[i * 2 + 1] = tph;
  }
  // assign magnitude and phase values from the accumulated buffer;
  for (int i = 0; i < n_fft; i++) {
    mdst[i * step] = buffer[i * 2];
    phdst[i * step] = buffer[i * 2 + 1];
  }
}

// Radix-2 FFT implementation - O(N log N) for power-of-2 sizes
static void radix2_fft(float *mdst, float *phdst, float *buffer, size_t n_fft,
                       size_t step) {
  if (n_fft == 1) {
    return;
  } else if (n_fft % 2 != 0) {
    // Fall back to DFT when we have a size that isn't factorable by 2
    simple_dft(mdst, phdst, buffer, n_fft, step);
    return;
  }

  radix2_fft(mdst, phdst, buffer, (size_t)n_fft / 2, step * 2);
  radix2_fft((float *)((char *)mdst + step * sizeof(float)),
             (float *)((char *)phdst + step * sizeof(float)), buffer,
             (size_t)n_fft / 2, step * 2);

  float km = M_PI * -2.0f / (float)n_fft;

  for (int i = 0; 2 * i < n_fft; i++) {
    float k = km * (float)i;
    float k1 = cosf(k);
    float k2 = sinf(k);

    float mp = mdst[i * 2 * step];
    float php = phdst[i * 2 * step];
    float mq = mdst[(i * 2 + 1) * step] * k1 - k2 * phdst[(i * 2 + 1) * step];
    float phq = mdst[(i * 2 + 1) * step] * k2 + k1 * phdst[(i * 2 + 1) * step];

    buffer[i + n_fft] = php + phq;
    buffer[i] = mp + mq;
    buffer[(i + (n_fft / 2)) + n_fft] = php - phq;
    buffer[(i + (n_fft / 2))] = mp - mq;
  }
  for (int i = 0; i < n_fft; i++) {
    mdst[i * step] = buffer[i];
    phdst[i * step] = buffer[i + n_fft];
  }
}

// STFT compute function parameters
struct stft_compute_params {
  int filter_length;
  int hop_length;
  bool compute_abs_and_angle;
  bool center;
};

// STFT computation function
static void ggml_compute_stft_f32(struct ggml_tensor *dst,
                                  const struct ggml_tensor *src0,
                                  const struct ggml_tensor *src1, int ith,
                                  int nth, void *userdata) {
  struct stft_compute_params *params = (struct stft_compute_params *)userdata;

  const float *w = (float *)src1->data;
  size_t n_fft = params->filter_length;
  size_t hop = params->hop_length;
  bool compute_abs_and_angle = params->compute_abs_and_angle;
  bool center = params->center;

  const int half = n_fft / 2;

  // Get tensor dimensions
  const int64_t ne00 = src0->ne[0]; // signal length
  const int64_t ne01 = src0->ne[1]; // batch size
  const int64_t ne1 = dst->ne[1];   // number of frames

  const size_t nb01 = src0->nb[1];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  // Allocate work buffer for FFT computation
  // Use maximum possible memory for the buffer rather than calculating the
  // largest uneven half
  float *buffer =
      (float *)malloc((n_fft * 2 + CACHE_LINE_SIZE_F32) * sizeof(float));

  // Zero the destination tensor
  memset(dst->data, 0.0f, ggml_nbytes(dst));

  // Process frames per thread
  const int hpt = (ne1 + nth - 1) / nth;
  const int ir0 = hpt * ith;
  const int ir1 = MIN(ir0 + hpt, ne1);

  for (int b = 0; b < ne01; b++) {
    for (int i1 = ir0; i1 < ir1; i1++) {
      const int ch = i1 * hop;
      float *mdst_data = (float *)((char *)dst->data + i1 * nb1 + b * nb2);
      float *phdst_data =
          (float *)((char *)dst->data + i1 * nb1 + nb3 + b * nb2);
      float *tgt_data = (float *)((char *)src0->data + b * nb01);

      // Pre-initialize the magnitude data with the window applied
      for (int i = 0; i < n_fft; i++) {
        int ai = center ? (ch - half + i) : (ch + i);

        // Handle reflective padding for center mode
        if (center) {
          if (ai < 0) {
            mdst_data[i] = tgt_data[-1 * ai] * w[i];
          } else if (ai >= ne00) {
            mdst_data[i] = tgt_data[ne00 - (ai - ne00 + 1)] * w[i];
          } else {
            mdst_data[i] = tgt_data[ai] * w[i];
          }
        } else {
          if (ai >= 0 && ai < ne00) {
            mdst_data[i] = tgt_data[ai] * w[i];
          } else {
            mdst_data[i] = 0.0f;
          }
        }
        phdst_data[i] = 0.0f; // Initialize phase to zero
      }

      // Perform FFT
      radix2_fft(mdst_data, phdst_data, buffer, n_fft, 1);

      // Convert to magnitude/phase if requested
      if (compute_abs_and_angle) {
        for (int i = 0; i < n_fft; i++) {
          float abs = sqrtf(mdst_data[i] * mdst_data[i] +
                            phdst_data[i] * phdst_data[i]);
          float agl = atan2f(phdst_data[i], mdst_data[i]);
          mdst_data[i] = abs;
          phdst_data[i] = agl;
        }
      }
    }
  }

  free(buffer);
}

// STFT implementation
struct ggml_tensor *stft(struct ggml_context *ctx, struct ggml_tensor *a,
                         struct ggml_tensor *window, int filter_length,
                         int hop_length, bool compute_abs_and_angle,
                         bool center) {

  // Calculate output dimensions exactly like original ggml
  const int64_t ne[4] = {
      filter_length, calculate_number_of_frames(a->ne[0], hop_length),
      a->ne[1], // batch dimension
      2         // real/imaginary or magnitude/phase
  };

  struct ggml_tensor *result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

  // Create parameters
  struct stft_compute_params *params =
      (struct stft_compute_params *)malloc(sizeof(struct stft_compute_params));
  params->filter_length = filter_length;
  params->hop_length = hop_length;
  params->compute_abs_and_angle = compute_abs_and_angle;
  params->center = center;

  return ggml_map_custom2(ctx, a, window, ggml_compute_stft_f32, 1, params);
}

// ISTFT compute function parameters
struct istft_compute_params {
  int filter_length;
  int hop_length;
  bool from_abs_and_angle;
  bool center;
};

// Proper inverse FFT implementation
static void radix2_ifft(float *mdst, float *phdst, float *buffer, float *tgt,
                        float *window, size_t n_fft, size_t step,
                        int min_length, int max_length, int index, int offset) {
  // For inverse FFT, we need to conjugate the input, apply forward FFT, then
  // conjugate and scale Conjugate the input (negate imaginary part)
  for (int i = 0; i < n_fft; i++) {
    phdst[i] = -phdst[i];
  }

  // Apply forward FFT
  radix2_fft(mdst, phdst, buffer, n_fft, step);

  // Conjugate the output and apply overlap-add with proper scaling
  for (int i = 0; i < n_fft; i++) {
    int base_index = i;                // Don't reverse for proper IFFT
    float real_part = mdst[i] / n_fft; // Scale by 1/N for IFFT
    float w = window[base_index];
    int tgt_index = base_index - offset;
    int location = index + tgt_index;
    if (location >= min_length && location < max_length) {
      tgt[location] += real_part * w;
    }
  }
}

// ISTFT computation function
static void ggml_compute_istft_f32(struct ggml_tensor *dst,
                                   const struct ggml_tensor *src0,
                                   const struct ggml_tensor *src1,
                                   const struct ggml_tensor *src2, int ith,
                                   int nth, void *userdata) {
  struct istft_compute_params *params = (struct istft_compute_params *)userdata;

  const float *window = (float *)src2->data;
  size_t n_fft = params->filter_length;
  size_t hop = params->hop_length;
  bool from_abs_and_angle = params->from_abs_and_angle;
  bool center = params->center;

  const int half = n_fft / 2;

  // Get tensor dimensions
  const int64_t ne1 = src0->ne[1]; // number of frames
  const int64_t ne2 = src0->ne[2]; // batch size
  const int64_t ne0 = dst->ne[0];  // output signal length

  const size_t nb1 = src0->nb[1];
  const size_t nb2 = src0->nb[2];
  const size_t nb3 = src0->nb[3];
  const size_t nb01 = dst->nb[1];

  // Allocate work buffer
  float *buffer =
      (float *)malloc((n_fft * 2 + CACHE_LINE_SIZE_F32) * sizeof(float));
  float *frame_buffer = (float *)malloc(n_fft * sizeof(float));
  float *phase_buffer = (float *)malloc(n_fft * sizeof(float));

  // Zero the destination tensor
  memset(dst->data, 0.0f, ggml_nbytes(dst));

  // Process batches per thread
  const int bpt = (ne2 + nth - 1) / nth;
  const int ib0 = bpt * ith;
  const int ib1 = MIN(ib0 + bpt, ne2);

  for (int b = ib0; b < ib1; b++) {
    float *tgt_data = (float *)((char *)dst->data + b * nb01);

    for (int i1 = 0; i1 < ne1; i1++) {
      const int ch = i1 * hop;
      float *msrc_data = (float *)((char *)src0->data + i1 * nb1 + b * nb2);
      float *phsrc_data =
          (float *)((char *)src0->data + i1 * nb1 + nb3 + b * nb2);

      // Copy frame data to work buffers
      for (int i = 0; i < n_fft; i++) {
        frame_buffer[i] = msrc_data[i];
        phase_buffer[i] = phsrc_data[i];
      }

      // Convert from magnitude/phase if needed
      if (from_abs_and_angle) {
        for (int i = 0; i < n_fft; i++) {
          float magnitude = frame_buffer[i];
          float phase = phase_buffer[i];
          frame_buffer[i] = magnitude * cosf(phase);
          phase_buffer[i] = magnitude * sinf(phase);
        }
      }

      // Perform inverse FFT with overlap-add
      int offset = center ? half : 0;
      int index = center ? (ch - half) : ch;
      radix2_ifft(frame_buffer, phase_buffer, buffer, tgt_data, (float *)window,
                  n_fft, 1, 0, ne0, index, offset);
    }
  }

  free(buffer);
  free(frame_buffer);
  free(phase_buffer);
}

// ISTFT implementation
struct ggml_tensor *istft(struct ggml_context *ctx, struct ggml_tensor *a,
                          struct ggml_tensor *window_squared_sum,
                          struct ggml_tensor *window, int filter_length,
                          int hop_length, bool from_abs_and_angle,
                          bool center) {

  // Calculate output dimensions exactly like original ggml
  const int64_t ne[4] = {calculate_original_length(a->ne[1], hop_length),
                         a->ne[2], // batch dimension
                         1, 1};

  struct ggml_tensor *result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

  // Create parameters
  struct istft_compute_params *params = (struct istft_compute_params *)malloc(
      sizeof(struct istft_compute_params));
  params->filter_length = filter_length;
  params->hop_length = hop_length;
  params->from_abs_and_angle = from_abs_and_angle;
  params->center = center;

  return ggml_map_custom3(ctx, a, window_squared_sum, window,
                          ggml_compute_istft_f32, 1, params);
}