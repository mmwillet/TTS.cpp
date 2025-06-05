#pragma once

#include "ggml.h"
#include "common.h"

struct quantization_params {
    uint32_t n_threads;
    ggml_type quantize_type;
    bool quantize_output_heads;
    bool quantize_text_embeddings;
    bool quantize_cross_attn_kv;
    bool convert_dac_to_f16;
    bool convert_non_quantizable_to_f16;
};

void quantize_gguf(str ifile, str ofile, const quantization_params & params);
