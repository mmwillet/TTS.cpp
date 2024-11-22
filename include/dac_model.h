#ifndef dac_model_h
#define dac_model_h

#include "util.h"

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
struct dac_model {
    // it doesn't make sense to expect gguf constant configuration to store information on the number of ggml tensors
    // as the gguf file is compiled from a pytorch model. This should be changed along with the load pattern.
    uint32_t n_tensors = 620;
    
    // These configs  are essentially built for the 44khZ 8kbps standard DAC model audio encoder and decoder
    uint32_t n_layers = 4;
    uint32_t n_heads = 9;
    uint32_t up_sampling_factor = 512;
    uint32_t max_generation_size = 2580;
    
    // this is the current byte offset into the model's buffer.
    size_t offset = 0;
    
    struct ggml_tensor * in_conv_kernel;
    struct ggml_tensor * in_conv_bias;
    struct ggml_tensor * out_conv_kernel;
    struct ggml_tensor * out_conv_bias;
    struct ggml_tensor * snake_alpha;
    std::vector<dac_layer> layers;
    std::vector<dac_quantize_layer> quantizer_layers;
    
    ggml_backend_buffer_type_t buffer = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    
    struct ggml_context * ctx;
    
    void prep_layers(gguf_context * meta);
    void prep_buffers_and_context(bool cpu_only, ggml_context * load_context);
    void prep_constants(gguf_context * meta);
    void setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only);
    void set_tensor(struct ggml_tensor * tensor, struct ggml_tensor * target);
    size_t max_nodes();
    void free();
};

#endif
