#pragma once

#include "../loaders.h"

extern const struct dummy_model_loader final : tts_model_loader {
    explicit dummy_model_loader();

    unique_ptr<tts_generation_runner> from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads,
                                                bool cpu_only, const generation_configuration & config) const override;
} dummy_loader;

class dummy_runner : public test_tts_generation_runner {
    unique_ptr<float[]> outputs{};
public:
    explicit dummy_runner() : test_tts_generation_runner{ dummy_loader } {}

    void generate(const char * sentence, tts_response & output, const generation_configuration & config) override;
};
