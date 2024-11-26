#ifndef sampler_h
#define sampler_h

#include <stdint.h>
#include <vector>
#include <random>

// currently this is only built to support single sequence output sampling without
// clustering, or beam search. While use of temperature is functional the repetition penalty is implemented
// only for single sequence rather than channel sequences (which means that it is rarely applicable
// for TTS generation.
struct sampler {
    // These default configurations are based on the generation configuration for Parler TTS Mini (version 1.0)
    uint32_t n_output_heads = 9;
    uint32_t eos_token_id = 1024;
    uint32_t vocab_size = 1088;
    float temperature = 1.0f;
    float repetition_penalty = 1.0f;
    std::vector<int32_t> last_token_ids;
    std::vector<uint32_t> repetition_counts;
    bool do_sample = true;
    bool apply_softmax = true;
    
    void sample(float * logits, std::vector<uint32_t> & output_tokens);
    void softmax(float * logits);
    void max(float * logits, std::vector<uint32_t> & output_tokens);
    void reset();
};

#endif
