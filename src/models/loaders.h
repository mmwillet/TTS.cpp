#pragma once

#include "../../include/common.h"

struct gguf_context;

struct tts_model_loader {
    /// Installs a model loader for the specified model architecture name
    explicit tts_model_loader(const char * arch, bool is_test = false);
    const char * const                        arch;
    const bool                                is_test;
    virtual unique_ptr<tts_generation_runner> from_file(
        gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads, bool cpu_only,
        /* TODO move to generate() */ const generation_configuration & config) const = 0;
  protected:
    ~tts_model_loader() = default;
};

unique_ptr<tts_generation_runner> runner_from_file(const char * fname, int n_threads,
                                                   const generation_configuration & config, bool cpu_only = true);
