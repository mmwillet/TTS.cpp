#pragma once

#include "ggml.h"
#include "common.h"

struct quantization_params {
    quantization_params(uint32_t n_threads, ggml_type quantize_type): n_threads(n_threads), quantize_type(quantize_type) {};
    uint32_t n_threads;
    ggml_type quantize_type; // quantization type
    bool quantize_output_heads = false;
    bool quantize_text_embeddings = false;
    bool quantize_cross_attn_kv = false;
    bool convert_dac_to_f16 = false;
    bool convert_non_quantizable_to_f16 = false;
};

void quantize_gguf(const std::string & ifile, const std::string & ofile, quantization_params * params);
