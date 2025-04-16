#ifndef sampler_h
#define sampler_h

#include <stdint.h>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>

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
    uint32_t top_k = 0;
    float repetition_penalty = 1.0f;
    std::vector<int32_t> last_token_ids;
    std::vector<uint32_t> repetition_counts;
    bool do_sample = true;
    bool apply_softmax = true;
    
    // these two settings allow the sampler to prefer eos tokens when their probability reaches a threshold
    bool eos_prioritized_processing = false;
    float prioritize_eos_threshold = 0.0f;
    
    void sample(float * logits, std::vector<uint32_t> & output_tokens);
    void softmax(float * logits, std::vector<std::vector<size_t>> picks, std::vector<uint32_t> max_indices);
    void max(float * logits, std::vector<uint32_t> & output_tokens);
    std::vector<std::vector<size_t>> topk(float* logits);
    void reset();
};

#endif
