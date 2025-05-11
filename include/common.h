#pragma once

#include <cstdint>
#include <string>
#include <map>
#include <memory>
#include <vector>
#include "ggml-cpp.h"

typedef const char * str;
using namespace std;
using namespace std::string_view_literals;
typedef std::string_view sv;

// Using this simple struct as opposed to a common std::vector allows us to return the cpu buffer
// pointer directly rather than copying the contents of the buffer to a predefined std::vector.
struct tts_response {
    float * data;
    size_t n_outputs = 0;
    uint32_t hidden_size; // this parameter is only currently used by the t5_encoder for which n_outputs corresponds to sequence length;
};

struct generation_configuration {
    generation_configuration(
        std::string voice = "",
        int top_k = 50, 
        float temperature = 1.0, 
        float repetition_penalty = 1.0, 
        bool use_cross_attn = true, 
        std::string espeak_voice_id = "",
        int max_tokens = 0,
        bool sample = true): top_k(top_k), temperature(temperature), repetition_penalty(repetition_penalty), use_cross_attn(use_cross_attn), sample(sample), voice(voice), espeak_voice_id(espeak_voice_id), max_tokens(max_tokens) {};

    bool use_cross_attn;
    float temperature;
    float repetition_penalty;
    int top_k;
    int max_tokens;
    std::string voice = "";
    bool sample = true;
    std::string espeak_voice_id = "";
};

struct tts_model;

struct tts_runner {
    using model_type = tts_model;
    explicit tts_runner(float sampling_rate = 44100.0f) : sampling_rate{sampling_rate} {}
    const float sampling_rate;
    virtual void prepare_post_load() = 0;
    virtual void update_conditional_prompt(str file_path, str prompt, int n_threads, bool cpu_only = true) {};
    virtual int generate(str prompt, tts_response * response) = 0;
};
