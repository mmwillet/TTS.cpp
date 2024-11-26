#include "sampler.h"

void sampler::sample(float * logits, std::vector<uint32_t> & output_tokens) {
    // assume that we are pointing to the start of the first token output;
    if (!do_sample) {
        return max(logits, output_tokens);
    }
    if (apply_softmax) {
        softmax(logits);
    }
    bool has_repetition_penalty = repetition_penalty != 1.0;
    if (has_repetition_penalty && (last_token_ids.size() == 0 || repetition_counts.size() == 0)) {
        reset();
    }
    std::minstd_rand gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n_output_heads; i++) {
        float assignment = dist(gen);
        float cumulative = 0.0f;
        for (uint32_t ii = 0; ii < vocab_size; ii++) {
            cumulative += *(logits+i*vocab_size+ii);
            if ((assignment <= cumulative) || (ii >= vocab_size + 1)) {
                if (has_repetition_penalty) {
                    if (last_token_ids[i] != ii) {
                        repetition_counts[i] = 0;
                    }
                    last_token_ids[i] = ii;
                    repetition_counts[i] += 1;
                }
                output_tokens.push_back(ii);
                break;
            }
        }
    }
}

void sampler::reset() {
    if (repetition_penalty != 1.0) {
        last_token_ids.clear();
        repetition_counts.clear();
        for (int i = 0; i < n_output_heads; i++) {
            last_token_ids.push_back(-1);
            repetition_counts.push_back(0);
        }
    }
}

void sampler::softmax(float * logits) {
    bool has_repetition_penalty = repetition_penalty != 1.0f;
    bool has_temperature = temperature != 1.0f;
    for (int i = 0; i < n_output_heads; i++) {
        float cumsum = 0.0f;
        for (int ii = 0; ii < vocab_size; ii++) {
            int index = i * vocab_size + ii;
            float v = *(logits + index);
            if (has_repetition_penalty && last_token_ids[i] == ii) {
                v /= (pow(repetition_penalty, repetition_counts[i]));
            }
            if (has_temperature) {
                v /= temperature;
            }
            v = expf(v);
            cumsum += v;
            logits[index] = v;
        }
        for (int ii = 0; ii < vocab_size; ii++) {
            int index = i * vocab_size + ii;
            float v = *(logits + index);
            logits[index] = v / cumsum;
        }
    }
}

void sampler::max(float * logits, std::vector<uint32_t> & output_tokens) {
    for (int i = 0; i < n_output_heads; i++) {
        float max = -INFINITY;
        uint32_t token_id = 0;
        for (uint32_t ii = 0; ii < vocab_size; ii++) {
            float v = *(logits+i*vocab_size+ii);
            if (v > max) {
                max = v;
                token_id = ii;
            }
        }
        output_tokens.push_back(token_id);
    }
}
