#pragma once

#include <utility>

#include "ggml.h"

class gguf_key_iterator {
    const gguf_context * const ctx;
    const int                  n_kv;
    int                        i{};

  public:
    explicit gguf_key_iterator(const gguf_context & ctx) : ctx{ &ctx }, n_kv{ gguf_get_n_kv(&ctx) } {}

    std::pair<int, const char *> operator*() const { return { i, gguf_get_key(ctx, i) }; }

    gguf_key_iterator & operator++() {
        ++i;
        return *this;
    }

    gguf_key_iterator begin() const {
        auto result{ *this };
        result.i = 0;
        return result;
    }

    gguf_key_iterator end() const {
        auto result{ *this };
        result.i = n_kv;
        return result;
    }

    bool operator==(const gguf_key_iterator &) const = default;
};

class ggml_tensor_iterator {
    const ggml_context * const ctx;
    ggml_tensor *              cur;

  public:
    explicit ggml_tensor_iterator(const ggml_context & ctx) : ctx{ &ctx }, cur{ ggml_get_first_tensor(&ctx) } {}

    ggml_tensor & operator*() const { return *cur; }

    ggml_tensor_iterator & operator++() {
        cur = ggml_get_next_tensor(ctx, cur);
        return *this;
    }

    ggml_tensor_iterator begin() const {
        auto result{ *this };
        result.cur = ggml_get_first_tensor(ctx);
        return result;
    }

    ggml_tensor_iterator end() const {
        auto result{ *this };
        result.cur = nullptr;
        return result;
    }

    bool operator==(const ggml_tensor_iterator &) const = default;
};
