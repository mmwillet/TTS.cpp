#ifndef dac_model_h
#define dac_model_h

#include "tts_model.h"

struct dac_residual_unit {
    struct ggml_tensor * in_snake_alpha;
    struct ggml_tensor * in_conv_kernel;
    struct ggml_tensor * in_conv_bias;
    struct ggml_tensor * out_snake_alpha;
    struct ggml_tensor * out_conv_kernel;
    struct ggml_tensor * out_conv_bias;
};

struct dac_layer {
    struct ggml_tensor * snake_alpha_in;
    struct ggml_tensor * out_conv_kernel;
    struct ggml_tensor * out_conv_bias;

    uint32_t padding;
    uint32_t stride;
    
    std::vector<dac_residual_unit> residual_blocks;
};

struct dac_quantize_layer {
    struct ggml_tensor * out_proj_kernel;
    struct ggml_tensor * out_proj_bias;
    struct ggml_tensor * codebook;
};

// this struct maintains the static tensors for the dac audio decoder graph.
// As such, this is designed to contain basic configuration and ggml tensor support for DAC.
// The dac_runner describes how the graph is built and run.
struct dac_model : tts_model {    
    // These configs  are essentially built for the 44khZ 8kbps standard DAC model audio encoder and decoder
    uint32_t n_layers = 4;
    uint32_t n_heads = 9;
    uint32_t up_sampling_factor = 512;
    uint32_t max_generation_size = 2580;
    
    struct ggml_tensor * in_conv_kernel;
    struct ggml_tensor * in_conv_bias;
    struct ggml_tensor * out_conv_kernel;
    struct ggml_tensor * out_conv_bias;
    struct ggml_tensor * snake_alpha;
    std::vector<dac_layer> layers;
    std::vector<dac_quantize_layer> quantizer_layers;
    void prep_constants(gguf_context * meta);
    void prep_layers(gguf_context * meta);
    void setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only) {
        prep_constants(meta_ctx);
        prep_layers(meta_ctx);
        tts_model::setup_from_file(meta_ctx, load_context, cpu_only, "audio_encoder");
    }
};

#endif
