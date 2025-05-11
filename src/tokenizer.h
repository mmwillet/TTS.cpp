#ifndef tokenizer_h
#define tokenizer_h

#include <unordered_map>
#include <stdint.h>
#include <map>
#include <unordered_set>
#include <regex>
#include "util.h"

struct token_trie {
    bool has_value{};
    uint32_t token;
    map<char, struct token_trie> children;
    
    void add(const string & gram, uint32_t token);
    void _add(const string & gram, uint32_t new_token, size_t index);
    const struct token_trie * traverse(const char c) const;
};

static regex duped_spaces("\\s{2,}");
static regex spaces("\\s");

struct result {
    uint32_t token;
    size_t offset;
    float score;
};

// much of this is implemented in llama.cpp, but in order to simplify this for my use case, I reimplementing here.
// There are several important simplifications here:
// 1. I only implement unigram tokenization
// 2. I don't need to support detokenization
struct unigram_tokenizer {
    explicit unigram_tokenizer(gguf_context * meta);
    unordered_map<string, uint32_t> vocab;
    vector<float> scores;
    token_trie root_trie;
    uint32_t unk_token;
    float unk_token_score;
    uint32_t eos_token = 1;
    bool dedupe_spaces = true;
    void tokenize(const string & text, vector<uint32_t> & tokens);
};

// While this functions like a tokenizer, no token ids are assigned as the token ids never need to be used in the context in which this is
// currently being used. This tokenizer pattern is currently being used by the phonemizer to break up a word into its relevant graphemes. 
// As such, only the graphemes need to be returned.
struct single_pass_tokenizer {
    single_pass_tokenizer(vector<string> tkns): tokens(tkns) {
        max_size = 0;
        for (auto token : tkns) {
            token_vocab.insert(token);
            if (token.size() > max_size) {
                max_size = token.size();
            }
        }
    }
    size_t max_size;
    uint32_t unknown_id = 0;
    vector<string> tokens;
    unordered_set<string> token_vocab;
    void tokenize(const string & text, vector<uint32_t> & token_ids);
    void token_split(const string & text, vector<string> & tokens);
};

single_pass_tokenizer * single_pass_tokenizer_from_gguf(gguf_context * meta, string key_name = "phonemizer.graphemes");

#endif
