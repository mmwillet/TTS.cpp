#ifndef parler_model_h
#define parler_model_h

#include "dac_model.h"
#include "t5_encoder_model.h"

struct parler_layer {
    struct ggml_tensor * self_attn_k_proj;
    struct ggml_tensor * self_attn_q_proj;
    struct ggml_tensor * self_attn_v_proj;
    struct ggml_tensor * self_attn_o_proj;
    struct ggml_tensor * self_attn_norm;
    struct ggml_tensor * self_attn_norm_bias;
    
    struct ggml_tensor * attn_k_proj;
    struct ggml_tensor * attn_q_proj;
    struct ggml_tensor * attn_v_proj;
    struct ggml_tensor * attn_o_proj;
    struct ggml_tensor * attn_norm;
    struct ggml_tensor * attn_norm_bias;
    
    struct ggml_tensor * cross_k;
    struct ggml_tensor * cross_v;
    
    struct ggml_tensor * fc1;
    struct ggml_tensor * fc2;
    struct ggml_tensor * final_norm;
    struct ggml_tensor * final_norm_bias;
};

struct parler_tts_model : tts_model {
    // These default configurations are based on the configuration of Parler TTS Mini (version 1.0)
    uint32_t n_output_heads = 9;
    uint32_t n_encode_length;
    uint32_t hidden_size = 1024;
    uint32_t max_ctx_length = 4096;
    uint32_t n_attn_heads = 16;
    uint32_t head_size = 64;
    uint32_t output_vocab_size = 1088;
    uint32_t eos_token_id = 1024;
    uint32_t audio_vocab_size = 1024;
    uint32_t max_generation_size = 2580;
    uint32_t n_layers = 24;
    uint32_t bos_token_id = 1025;
    uint32_t max_cross_nodes = 32;
    uint32_t prompt_vocab_size;

    bool use_cross_attn = true;
    
    std::vector<struct ggml_tensor*> embds;
    std::vector<parler_layer*> layers;
    std::vector<struct ggml_tensor*> heads;
    
    struct ggml_tensor * precomputed_input_emb;
    struct ggml_tensor * precomputed_positional_embds;
    
    struct ggml_tensor * layer_norm;
    struct ggml_tensor * layer_norm_bias;
    struct ggml_tensor * prompt_embd;
    
    void prep_constants(gguf_context * meta);
    void prep_layers(gguf_context * meta);
    void prep_cross_key_values(struct t5_response * conditional_prompt = nullptr);
    void setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only) {
        prep_constants(meta_ctx);
        prep_layers(meta_ctx);
        tts_model::setup_from_file(meta_ctx, load_context, cpu_only, "decoder", 1.25, n_encode_length*hidden_size*sizeof(float)*n_layers*2);
    }
};

#endif
