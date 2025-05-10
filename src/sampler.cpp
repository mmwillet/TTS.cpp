#include "sampler.h"

void sampler::sample(float * logits, std::vector<uint32_t> & output_tokens) {\
    // assume that we are pointing to the start of the first token output;
    if (!do_sample) {
        return max(logits, output_tokens);
    }
    std::vector<uint32_t> max_vals;

    // This allows us to perform an effective softmax without logarithms or big number calculations.
    // Additionally by avoiding large number division we drastically improve the stability of
    // our softmax implementation;
    max(logits, max_vals);

    std::vector<std::vector<size_t>> picks;
    bool use_nucleus_sampling = false;
    if (top_k > 0 && top_k < vocab_size) {
        picks = topk(logits);
        use_nucleus_sampling = true;
    }

    softmax(logits, picks, max_vals);
    bool has_repetition_penalty = repetition_penalty != 1.0;
    if (has_repetition_penalty && (last_token_ids.size() == 0 || repetition_counts.size() == 0)) {
        reset();
    }
    std::minstd_rand gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n_output_heads; i++) {
        float assignment = dist(gen);
        float cumulative = 0.0f;
        for (uint32_t j = 0; j < (use_nucleus_sampling ? top_k : vocab_size); j++) {
            int ii = use_nucleus_sampling ? (int) picks[i][j] : j;
            cumulative += *(logits+(i*vocab_size+ii));
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

void sampler::softmax(float * logits, std::vector<std::vector<size_t>> picks, std::vector<uint32_t> max_indices) {
    bool use_nucleus_sampling = picks.size() > 0;
    bool has_repetition_penalty = repetition_penalty != 1.0f;
    bool has_temperature = temperature != 1.0f;
    for (int i = 0; i < n_output_heads; i++) {
        float cumsum = 0.0;
        float max_val = logits[i*vocab_size + max_indices[i]];
        if (has_repetition_penalty && last_token_ids[i] == max_indices[i]) {
            max_val /= (pow(repetition_penalty, repetition_counts[i]));
        }
        if (has_temperature) {
            max_val /= temperature;
        }
        for (int j = 0; j < (use_nucleus_sampling ? picks[i].size() : vocab_size); j++) {
            int ii = use_nucleus_sampling ? (int) picks[i][j] : j;
            int index = i * vocab_size + ii;
            float v = *(logits + index);
            if (has_repetition_penalty && last_token_ids[i] == ii) {
                v /= (pow(repetition_penalty, repetition_counts[i]));
            }
            if (has_temperature) {
                v /= temperature;
            }
            v = expf(v - max_val);
            cumsum += v;
            logits[index] = v;
        }
        for (int j = 0; j < (use_nucleus_sampling ? picks[i].size() : vocab_size); j++) {
            int ii = use_nucleus_sampling ? picks[i][j] : j;
            int index = i * vocab_size + ii;
            float v = *(logits + index);
            logits[index] = v / cumsum;
        }
    }
}

std::vector<std::vector<size_t>> sampler::topk(float* logits) {
    bool has_repetition_penalty = repetition_penalty != 1.0f;
    std::vector<std::vector<size_t>> head_picks;
    if (vocab_size < top_k) {
        // technically we should never get here, but lets be protective.
        for (int i = 0; i < n_output_heads; i++) {
            std::vector<size_t> picks(vocab_size);
            iota(picks.begin(), picks.end(), 0);
            head_picks.push_back(picks);
        }
        return head_picks;
    }
    for (int i = 0; i < n_output_heads; i++) {
        std::vector<size_t> picks(vocab_size);
        iota(picks.begin(), picks.end(), 0);
        // have to sort with repetition penalty applied so not to inavertently trim our nucleus size.
        std::sort(picks.begin(), picks.end(), [&logits, &i, &has_repetition_penalty, this](size_t s1, size_t s2) {
            float v1 = logits[i*vocab_size+s1];
            float v2 = logits[i*vocab_size+s2];
            if (has_repetition_penalty && last_token_ids[i] == s1) {
                v1 /= (pow(repetition_penalty, repetition_counts[i]));
            } else if (has_repetition_penalty && last_token_ids[i] == s2) {
                v2 /= (pow(repetition_penalty, repetition_counts[i]));
            }
            return v1 < v2;
        });
        head_picks.push_back(std::vector<size_t>(picks.end() - top_k, picks.end()));
    }
    return head_picks;
}

void sampler::max(float * logits, std::vector<uint32_t> & output_tokens) {
    bool has_repetition_penalty = repetition_penalty != 1.0f;
    for (int i = 0; i < n_output_heads; i++) {
        float max = -INFINITY;
        uint32_t token_id = 0;
        for (uint32_t ii = 0; ii < vocab_size; ii++) {
            float v = *(logits+i*vocab_size+ii);
            // while repetition penalty will never be used for maximum token selection, it is used for the logarithmic stabilization of 
            // the softmax function in which case it is possible for repetition counts to be set.
            if (has_repetition_penalty && last_token_ids[i] == ii) {
                v /= (pow(repetition_penalty, repetition_counts[i]));
            }
            if (v > max) {
                max = v;
                token_id = ii;
            }
        }
        output_tokens.push_back(token_id);
    }
}
