#pragma once

#include "tts_model.h"

struct dac_residual_unit {
    ggml_tensor * in_snake_alpha;
    ggml_tensor * in_conv_kernel;
    ggml_tensor * in_conv_bias;
    ggml_tensor * out_snake_alpha;
    ggml_tensor * out_conv_kernel;
    ggml_tensor * out_conv_bias;

    void assign_weight(ggml_context * ctx, sv name, ggml_tensor *tensor);
};

struct dac_layer_constants {
    uint32_t padding;
    uint32_t stride;
};

struct dac_layer {
    ggml_tensor * snake_alpha_in;
    ggml_tensor * out_conv_kernel;
    ggml_tensor * out_conv_bias;

    dac_layer_constants * constants;

    array<dac_residual_unit, 3> residual_blocks;

    void assign_weight(ggml_context * ctx, sv name, ggml_tensor *tensor);
};

struct dac_quantize_layer {
    ggml_tensor * out_proj_kernel;
    ggml_tensor * out_proj_bias;
    ggml_tensor * codebook;

    void assign_weight(ggml_context * ctx, sv name, ggml_tensor *tensor);
};

struct dac_model_constants {
    explicit dac_model_constants(gguf_context *meta);

    // These configs are essentially built for the 44khZ 8kbps standard DAC model audio encoder and decoder
    static constexpr uint32_t n_layers{4};
    const uint32_t n_heads{9};
    const uint32_t up_sampling_factor{512};
    const uint32_t max_generation_size{2580};
    const vector<dac_layer_constants> layer_constants;
};

// this struct maintains the static tensors for the dac audio decoder graph.
// As such, this is designed to contain basic configuration and ggml tensor support for DAC.
// The dac_runner describes how the graph is built and run.
struct dac_model : tts_model, dac_model_constants {
    ggml_tensor * in_conv_kernel;
    ggml_tensor * in_conv_bias;
    ggml_tensor * out_conv_kernel;
    ggml_tensor * out_conv_bias;
    ggml_tensor * snake_alpha;
    vector<dac_layer> layers;
    vector<dac_quantize_layer> quantizer_layers;

    void assign_weight(sv name, ggml_tensor *weight) override;
    void prep_constants(gguf_context * meta);
    void prep_layers(gguf_context * meta);
    explicit dac_model(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only) :
            tts_model{}, dac_model_constants{meta_ctx}, layers(n_layers), quantizer_layers(n_heads) {
        prep_layers(meta_ctx);
        tts_model::setup_from_file(meta_ctx, load_context, cpu_only, "audio_encoder");
    }
};

// for loading DAC model from gguf file
void assign_residual_unit(dac_model * model, dac_residual_unit * layer, string name, ggml_tensor * tensor);
void assign_dac_layer(dac_model * model, dac_layer * layer, string name, ggml_tensor * tensor);
void assign_quantizer_layer(dac_model * model, dac_quantize_layer  layer, string name, ggml_tensor * tensor);

struct dac_ubatch {
    uint32_t * input_tokens;
    uint32_t sequence_length;
};

ggml_tensor * dac_build_audio_inputs(struct ggml_context * ctx, struct dac_context * dctx, const dac_ubatch & batch, vector<dac_quantize_layer> layers);
ggml_tensor * build_residual_unit(ggml_context * ctx, ggml_tensor * cur, dac_residual_unit & u, int padding, int dilation);
ggml_tensor * build_decoder_block(ggml_context * ctx, ggml_tensor * cur, dac_layer & l, struct dac_context * dctx);

struct dac_context : runner_context {

};

// This struct is intended to manage the dac model's graph compilation and compute function.
struct dac_runner : runner_context {
    explicit dac_runner(shared_ptr<dac_model> model): model{model} {};
    const shared_ptr<dac_model> model;
    ggml_tensor * inp_tokens;

    void prepare_post_load();
    ggml_cgraph * build_dac_graph(dac_ubatch & batch);
    void run(uint32_t * input_tokens, uint32_t sequence_length, struct tts_response * outputs);
};
