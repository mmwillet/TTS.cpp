#include "tokenizer.h"

void token_trie::add(const std::string & gram, uint32_t token) {
    _add(gram, token, 0);
}

void token_trie::_add(const std::string & gram, uint32_t new_token, size_t index) {
    if (index >= gram.size()) {
        has_value = true;
        token = new_token;
        return;
    }
    const char c = gram[index];
    auto res = children.find(c);
    if (res != children.end()) {
        res->second._add(gram, new_token, index + 1);
    } else {
        struct token_trie nt{};
        nt._add(gram, new_token, index + 1);
        children[c] = nt;
    }
}

const struct token_trie * token_trie::traverse(const char c) const {
    auto res = children.find(c);
    if (res != children.end()) {
        return &res->second;
    }

    return NULL;
}

size_t unicode_len_utf8(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

void unigram_tokenizer::initialize_tokenizer() {
    for (const auto it : vocab) {
        root_trie.add(it.first, it.second);
    }
    init = true;
}

// the general approach here is to find the character grams that sum to the max possible value over the entire text sequence.
// The particular algorithm used here effectively works by walking the text and at each index storing the max value of all possible gram combinations
// we can then reverse that sequence to pick the best possible tokens.
void unigram_tokenizer::tokenize(const std::string & text, std::vector<uint32_t> & tokens) {
    assert(init);
    // the parler tokenizer's normalizer (i.e. the bert normalizer implemented by huggingface tokenizers libs) only deduplicates and strips extra spaces and
    // optionally handles chinese characters and accents (neither of which are currently supported here).
    std::string normalized = text;
    if (dedupe_spaces) {
        normalized = " " + std::regex_replace(text, duped_spaces, " ");
    }
    
    size_t text_length = normalized.size();

    // initialize score_sum to neg infinity so it will be always lower than sums of token scores
    std::vector<struct result> results(text_length + 1, {unk_token, 0, -INFINITY});
    results[0] = { unk_token, 0, 0 };
    
    size_t offset = 0;

    while (offset < text_length) {
        size_t current_offset = offset;
        // pulled this directly from llama.cpp; I suspect that this is for handling of non-utf8 steps (to be marked as unknown tokens)
        size_t n_utf8_code_units = std::min<size_t>(unicode_len_utf8(normalized[offset]), text_length - offset);

        bool found_unknown = true;
        const struct result & current_best = results[offset];
        
        // find the current branch in the trie
        const struct token_trie * node = root_trie.traverse(normalized[current_offset++]);
        // search for the next token
        while (current_offset <= text_length && node != NULL) {
            // check if this is a complete token (it could just be an unkown step between two tokens).
            if (node->has_value) {
                // check if it corresponds to the whole utf8 step
                if (current_offset - offset == n_utf8_code_units) {
                    found_unknown = false;
                }
                float score = current_best.score + scores[node->token];
                struct result & current_champ = results[current_offset];
                if (score > current_champ.score) {
                    struct result challenger = { node->token, offset, score };
                    current_champ = challenger;
                }
            }
            node = node->traverse(normalized[current_offset++]);
        }

        // if we found an unknown token, process it
        if (found_unknown) {
            current_offset = offset + n_utf8_code_units;
            struct result & current_champ = results[current_offset];
            float score = current_best.score + unk_token_score;
            if (score > current_champ.score) {
                struct result challenger = { unk_token, offset, score };
                current_champ = challenger;
            }
        }

        // move one utf8 step
        offset += n_utf8_code_units;
    }

    // if we have more than on unknown token in a row, we can join them.
    bool is_prev_unknown = false;
    // iterate from the last result backwards and get the best performing tokens
    for (struct result & result = results[text_length]; ; result = results[result.offset]) {
        bool is_unknown = result.token == unk_token;
        if (!(is_prev_unknown && is_unknown)) {
            tokens.push_back(result.token);
        }
        if (result.offset == 0) {
            break;
        }
        is_prev_unknown = is_unknown;
    }

    // reverse the tokens since we added tokens starting from the end of the input
    std::reverse(tokens.begin(), tokens.end());
}
