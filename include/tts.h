#ifndef tts_h
#define tts_h

#include "parler_model.h"
#include <thread>
#include <fstream>

struct tts_runner * runner_from_file(const std::string & fname, int n_threads, generation_configuration * config, bool cpu_only = true);
int generate(tts_runner * runner, std::string sentence, struct tts_response * response, generation_configuration * config);
void update_conditional_prompt(tts_runner * runner, const std::string file_path, const std::string prompt, bool cpu_only = true);

struct quantization_params {
    quantization_params(uint32_t n_threads, enum ggml_type quantize_type, void * imatrix = nullptr): n_threads(n_threads), quantize_type(quantize_type), imatrix(imatrix) {};
    uint32_t n_threads;
    enum ggml_type quantize_type; // quantization type
    void * imatrix = nullptr; // pointer to importance matrix data
    bool quantize_output_heads = false;
    bool quantize_text_embeddings = false;
    bool quantize_cross_attn_kv = false;
    bool convert_dac_to_f16 = false;
};

void quantize_gguf(const std::string & ifile, const std::string & ofile, struct quantization_params * params);

#endif
