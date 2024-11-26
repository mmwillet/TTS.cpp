#ifndef tokenizer_h
#define tokenizer_h

#include <unordered_map>
#include <stdint.h>
#include <map>
#include <regex>
#include "util.h"


struct token_trie {
    bool has_value = false;
    uint32_t token;
    std::map<char, struct token_trie> children;
    
    void add(const std::string & gram, uint32_t token);
    void _add(const std::string & gram, uint32_t new_token, size_t index);
    const struct token_trie * traverse(const char c) const;
};

static std::regex duped_spaces("\\s{2,}");
static std::regex spaces("\\s");

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
    unigram_tokenizer(std::unordered_map<std::string, uint32_t> vocab, uint32_t unk_token, float unk_token_score, std::vector<float> scores): vocab(vocab), unk_token(unk_token), unk_token_score(unk_token_score), scores(scores) {};
    ~unigram_tokenizer() = default;
    
    std::unordered_map<std::string, uint32_t> vocab;
    std::vector<float> scores;
    struct token_trie root_trie;
    uint32_t unk_token;
    float unk_token_score;
    uint32_t eos_token = 1;
    bool dedupe_spaces = true;
    bool init = false;
    
    void initialize_tokenizer();
    void tokenize(const std::string & text, std::vector<uint32_t> & tokens);
};

#endif
