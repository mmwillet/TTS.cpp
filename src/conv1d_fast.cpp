#include <immintrin.h>

#include "ggml-impl.h"

// TODO the output is wrong

// TODO does the kernel need flipped across axis K?
#define K 11
// TODO convert to template parameter
#define IC 128
#define OC IC

[[gnu::target("avx2")]] void tts_compute_forward_conv_1d(ggml_tensor * dst, const ggml_tensor * kernel,
                                                         const ggml_tensor * data, int ith, int nth, void *) {
    const size_t nb11{data->nb[1]};
    const int L{static_cast<int>(data->ne[1])};
    GGML_ASSERT(L >= K);
    GGML_ASSERT(kernel->ne[0] == IC);
    GGML_ASSERT(kernel->ne[2] == OC);
    const int oc_per_thread{static_cast<int>(kernel->ne[2] / nth)};
    const int oc_start{oc_per_thread * ith};
    const int oc_end{static_cast<int>(ith == nth - 1 ? kernel->ne[2] : oc_start + oc_per_thread)};

    // Naïve matrix multiplication should be fine
    // The emphasis is on keeping oc_kernel in L1d cache
    const auto dot = [](int k, const float (&oc_kernel)[K][IC], const float (&now_)[IC]) -> float {
        const __m256 now = _mm256_load_ps(now_);
        // Like -ffast-math this ignores float non-commutativity
        __m256 sum_buffer_a{_mm256_setzero_ps()};
        __m256 sum_buffer_b{_mm256_setzero_ps()};
        static_assert(IC % 32 == 0);
        for (int ic{}; ic < IC; ic += /*32*/ 16) {
            // mm256 / float32 = 8
            sum_buffer_a = _mm256_fmadd_ps(_mm256_load_ps(&oc_kernel[k][ic]), now, sum_buffer_a);
            sum_buffer_b = _mm256_fmadd_ps(_mm256_load_ps(&oc_kernel[k][ic + 8]), now, sum_buffer_b);
        }
        sum_buffer_a = _mm256_add_ps(sum_buffer_a, sum_buffer_b);
        const __m128 sum_4{_mm_add_ps(_mm256_castps256_ps128(sum_buffer_a), _mm256_extractf128_ps(sum_buffer_a, 1))};
        // Don't use hadd like ggml does. It's slow: https://stackoverflow.com/a/49943540/10477326
        const __m128 sum_2{_mm_add_ps(sum_4, _mm_movehl_ps(sum_4, sum_4))};
        const __m128 sum{_mm_add_ps(sum_2, _mm_movehdup_ps(sum_2))};
        return _mm_cvtss_f32(sum);
    };
    // TODO fix -Wpointer-arith

    float ring_buffer[OC][K]{}; // TODO change to [K][4]
    int time{};
    // beginning
    for (; time < K >> 1; ++time) {
        const float (&now)[IC]{*static_cast<float(*)[IC]>(data->data + time * nb11)};
        const int k_start{(K >> 1) - time};
        for (int oc{oc_start}; oc < oc_end; ++oc) {
            const float (&oc_kernel)[K][IC]{*static_cast<float(*)[K][IC]>(kernel->data + oc * kernel->nb[2])};
            for (int k{k_start}; k < K; ++k) {
                ring_buffer[oc][k - k_start] += dot(K - k, oc_kernel, now);
            }
        }
    }
    int tail{};
    // middle
    for (; time < L; ++time) {
        const float (&now)[IC]{*static_cast<float(*)[IC]>(data->data + time * nb11)};
        for (int oc{oc_start}; oc < oc_end; ++oc) {
            float * oc_dst{static_cast<float *>(dst->data + oc * dst->nb[1])};
            const float (&oc_kernel)[K][IC]{*static_cast<float(*)[K][IC]>(kernel->data + oc * kernel->nb[2])};
            for (int k{}; k < K; ++k) {
                ring_buffer[oc][tail++] += dot(K - k, oc_kernel, now);
                tail = tail == K ? 0 : tail;
            }
            oc_dst[time - (K >> 1)] = ring_buffer[oc][tail];
        }
        ++tail;
        tail = tail == K ? 0 : tail;
    }
    // ending
    for (; time < L + (K >> 1); ++time) {
        for (int oc{oc_start}; oc < oc_end; ++oc) {
            float * oc_dst{static_cast<float *>(dst->data + oc * dst->nb[1])};
            oc_dst[time - (K >> 1)] = ring_buffer[oc][tail];
        }
        ++tail;
        tail = tail == K ? 0 : tail;
    }
}

// These non-intrinsics versions below seem 10x slower not faster

// void tts_compute_forward_conv_1d(ggml_tensor * dst, const ggml_tensor * kernel, const ggml_tensor * data,
//                                  int ith, int nth, void *) {
//     const size_t nb11{data->nb[1]};
//     const int L{static_cast<int>(data->ne[1])};
//     GGML_ASSERT(L >= K);
//     GGML_ASSERT(kernel->ne[0] == IC);
//     GGML_ASSERT(kernel->ne[2] == OC);
//     const int oc_per_thread{static_cast<int>(kernel->ne[2] / nth)};
//     const int oc_start{oc_per_thread * ith};
//     const int oc_end{static_cast<int>(ith == nth - 1 ? kernel->ne[2] : oc_start + oc_per_thread)};
//
//     const auto dot = [](int k, const float (&oc_kernel)[K][IC], const float (&now)[IC]) -> float {
//         // Naïve matrix multiplication should be fine
//         // The emphasis is on keeping oc_kernel in L1d cache
// #define SIMD_WIDTH 8 // 8 floats in 1 YMM register
//         static_assert((IC & (SIMD_WIDTH - 1)) == 0);
//         float sum_buffer[SIMD_WIDTH]{};
//         for (int ic{}; ic < IC; ++ic) {
//             float & sum_buffer_dst{sum_buffer[ic & (SIMD_WIDTH - 1)]};
//             sum_buffer_dst = fma(oc_kernel[k][ic], now[ic], sum_buffer_dst);
//         }
//         // TODO gcc seems to undo this
//         for (int simd_width{SIMD_WIDTH >> 1}; simd_width > 1; simd_width >>= 1) {
//             for (int i{}; i < simd_width; ++i) {
//                 sum_buffer[i] += sum_buffer[i + simd_width];
//             }
//         }
//         return sum_buffer[0];
//     };
//
//     float ring_buffer[OC][K]{};
//     int time{};
//     // beginning
//     for (; time < K >> 1; ++time) {
//         const float (&now)[IC]{*static_cast<float(*)[IC]>(data->data + time * nb11)};
//         const int k_start{(K >> 1) - time};
//         for (int oc{oc_start}; oc < oc_end; ++oc) {
//             const float (&oc_kernel)[K][IC]{*static_cast<float(*)[K][IC]>(kernel->data + oc * kernel->nb[2])};
//             for (int k{k_start}; k < K; ++k) {
//                 ring_buffer[oc][k - k_start] += dot(K - k, oc_kernel, now);
//             }
//         }
//     }
//     int tail{};
//     // middle
//     for (; time < L; ++time) {
//         const float (&now)[IC]{*static_cast<float(*)[IC]>(data->data + time * nb11)};
//         for (int oc{oc_start}; oc < oc_end; ++oc) {
//             float * oc_dst{static_cast<float *>(dst->data + oc * dst->nb[1])};
//             const float (&oc_kernel)[K][IC]{*static_cast<float(*)[K][IC]>(kernel->data + oc * kernel->nb[2])};
//             for (int k{}; k < K; ++k) {
//                 ring_buffer[oc][tail++] += dot(K - k, oc_kernel, now);
//                 tail = tail == K ? 0 : tail;
//             }
//             oc_dst[time - (K >> 1)] = ring_buffer[oc][tail];
//         }
//         ++tail;
//         tail = tail == K ? 0 : tail;
//     }
//     // ending
//     for (; time < L + (K >> 1); ++time) {
//         for (int oc{oc_start}; oc < oc_end; ++oc) {
//             float * oc_dst{static_cast<float *>(dst->data + oc * dst->nb[1])};
//             oc_dst[time - (K >> 1)] = ring_buffer[oc][tail];
//         }
//         ++tail;
//         tail = tail == K ? 0 : tail;
//     }
// }

// void tts_compute_forward_conv_1d(ggml_tensor * dst, const ggml_tensor * kernel, const ggml_tensor * data,
//                                         int ith, int nth, void *) {
//     const size_t nb11{data->nb[1]};
//     const int L{static_cast<int>(data->ne[1])};
//     GGML_ASSERT(L >= K);
//     const int oc_per_thread{static_cast<int>(kernel->ne[2] / nth)};
//     const int oc_start{oc_per_thread * ith};
//     const int oc_end{static_cast<int>(ith == nth - 1 ? kernel->ne[2] : oc_start + oc_per_thread)};
//
//     const auto dot = [](int k, const float (&oc_kernel)[K][IC], const float (&now)[IC]) -> float {
//         // Naïve matrix multiplication should be fine
//         // The emphasis is on keeping oc_kernel in L1d cache
// #define SIMD_WIDTH 8 // 8 floats in 1 YMM register
//         static_assert((IC & (SIMD_WIDTH - 1)) == 0);
//         float sum_buffer[SIMD_WIDTH]{};
//         for (int ic{}; ic < IC; ++ic) {
//             float & sum_buffer_dst{sum_buffer[ic & (SIMD_WIDTH - 1)]};
//             sum_buffer_dst = fma(oc_kernel[k][ic], now[ic], sum_buffer_dst);
//         }
//         // TODO gcc seems to undo this
//         for (int simd_width{SIMD_WIDTH >> 1}; simd_width > 1; simd_width >>= 1) {
//             for (int i{}; i < simd_width; ++i) {
//                 sum_buffer[i] += sum_buffer[i + simd_width];
//             }
//         }
//         return sum_buffer[0];
//     };
//
//     for (int oc{oc_start}; oc < oc_end; ++oc) {
//         float * oc_dst{static_cast<float *>(dst->data + oc * dst->nb[1])};
//         const float (&oc_kernel)[K][IC]{*static_cast<float(*)[K][IC]>(kernel->data + oc * kernel->nb[2])};
//         float ring_buffer[K]{};
//         int time{};
//         // beginning
//         for (; time < K >> 1; ++time) {
//             const float (&now)[IC]{*static_cast<float(*)[IC]>(data->data + time * nb11)};
//             const int k_start{(K >> 1) - time};
//             for (int k{k_start}; k < K; ++k) {
//                 ring_buffer[k - k_start] += dot(K - k, oc_kernel, now);
//             }
//         }
//         int tail{};
//         // middle
//         for (; time < L; ++time) {
//             const float (&now)[IC]{*static_cast<float(*)[IC]>(data->data + time * nb11)};
//             for (int k{}; k < K; ++k) {
//                 ring_buffer[tail++] += dot(K - k, oc_kernel, now);
//                 tail = tail == K ? 0 : tail;
//             }
//             oc_dst[time - (K >> 1)] = ring_buffer[tail++];
//             tail = tail == K ? 0 : tail;
//         }
//         // ending
//         for (; time < L + (K >> 1); ++time) {
//             oc_dst[time - (K >> 1)] = ring_buffer[tail++];
//             tail = tail == K ? 0 : tail;
//         }
//     }
// }

static ggml_tensor * tts_try_conv_1d(ggml_context * ctx, ggml_tensor * kernel, ggml_tensor * data) {
    // ggml works fine if this can stay in the 32KiB L1d cache
    if (data->ne[0] < 64) {
        return nullptr;
    }
    if (kernel->ne[2] != 128 || data->ne[1] != 128) {
        return nullptr; // Not specialized for yet
    }
    // if (true) {
    //     return nullptr; // Disable the new code for comparison during debugging
    // }
    kernel = ggml_cont(ctx, ggml_transpose(ctx, kernel)); // [OC, IC, K] => [OC, K, IC]
    data = ggml_cont(ctx, ggml_transpose(ctx, data)); // [IC, L] => [L, IC]

    ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, data->ne[1], kernel->ne[2]); // [OC, L]

    static constexpr ggml_map_custom2_op_params params{
        .fun{tts_compute_forward_conv_1d},
        .n_tasks{GGML_N_TASKS_MAX},
        .userdata{},
    };
    ggml_set_op_params(result, &params, sizeof(params));

    result->op = GGML_OP_MAP_CUSTOM2;
    result->src[0] = kernel;
    result->src[1] = data;

    return result;
}

ggml_tensor * tts_conv_1d(ggml_context * ctx, ggml_tensor * kernel, ggml_tensor * data, int s0, int p0, int d0) {
    // GGML_ASSERT(d0 == 1);
    GGML_ASSERT(s0 == 1);
    GGML_ASSERT(kernel->ne[0] & 1 == 1);
    GGML_ASSERT((kernel->ne[0] >> 1) * d0 == p0); // mode="same", L = IL = OL
    GGML_ASSERT(kernel->ne[1] == data->ne[1]);
    GGML_ASSERT(kernel->ne[3] == 1); // [OC, IC, K]
    GGML_ASSERT(data->ne[2] == 1 && kernel->ne[3] == 1);
    GGML_ASSERT(kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(data->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(kernel));
    // GGML_ASSERT(ggml_is_contiguous(data));
    ggml_tensor * result{};
#ifndef GGML_USE_METAL
    if (d0 == 1) {
        // TODO d0 == 3, d0 == 5
        result = tts_try_conv_1d(ctx, kernel, data);
    }
#endif
    if (!result) {
        result = ggml_conv_1d(ctx, kernel, data, 1, p0, d0);
    }
    // result = ggml_cont(ctx, ggml_transpose(ctx, result)); // [OC, L] => [L, OC]
    return result;
}
