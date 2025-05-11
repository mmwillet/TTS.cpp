#pragma once

#include <fstream>
#include <thread>
#include "common.h"

unique_ptr<tts_runner> runner_from_file(str fname, int n_threads, generation_configuration * config, bool cpu_only = true);
int generate(tts_runner * runner, string sentence, struct tts_response * response, generation_configuration * config);
void update_conditional_prompt(tts_runner * runner, const string file_path, const string prompt, bool cpu_only = true);

struct quantization_params {
    quantization_params(uint32_t n_threads, enum ggml_type quantize_type): n_threads(n_threads), quantize_type(quantize_type) {};
    uint32_t n_threads;
    enum ggml_type quantize_type; // quantization type
    bool quantize_output_heads = false;
    bool quantize_text_embeddings = false;
    bool quantize_cross_attn_kv = false;
    bool convert_dac_to_f16 = false;
};

void quantize_gguf(str ifile, str ofile, const quantization_params & params);
