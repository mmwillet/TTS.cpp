#pragma once

#include "dac_model.h"
#include "sampler.h"

struct dia_encoder_layer {
    ggml_tensor * k;
    ggml_tensor * q;
    ggml_tensor * v;
    ggml_tensor * o;
    ggml_tensor * self_attn_norm;

    ggml_tensor * gate;
    ggml_tensor * up;
    ggml_tensor * out;
    ggml_tensor * mlp_norm;
};

struct dia_decoder_layer {
    ggml_tensor * self_attn_k;
    ggml_tensor * self_attn_q;
    ggml_tensor * self_attn_v;
    ggml_tensor * self_attn_o;
    ggml_tensor * self_attn_norm;

    ggml_tensor * cross_attn_k;
    ggml_tensor * cross_attn_q;
    ggml_tensor * cross_attn_v;
    ggml_tensor * cross_attn_o;
    ggml_tensor * cross_attn_norm;

    ggml_tensor * gate;
    ggml_tensor * up;
    ggml_tensor * out;
    ggml_tensor * mlp_norm;
};

struct dia_encoder {
    explicit dia_encoder(size_t n_encoder_layers);

    ggml_tensor * norm;
    ggml_tensor * embedding;
    vector<dia_encoder_layer> layers;
};

struct dia_decoder {
    explicit dia_decoder(size_t n_decoder_layers, size_t n_output_heads);

    ggml_tensor * norm;
    vector<ggml_tensor *> embds;
    vector<ggml_tensor *> heads;
    vector<dia_decoder_layer> layers;
};

struct dia_model_constants {
    explicit dia_model_constants(gguf_context * meta);

    // These default configurations are based on the default configuration for the Dia 1.68b param model.
    const uint32_t n_output_heads{9};
    const uint32_t n_encoder_layers{12};
    const uint32_t n_decoder_layers{18};
    const uint32_t encoder_hidden_size{1024};
    const uint32_t decoder_hidden_size{2048};
    const uint32_t encoder_attn_heads{16};
    const uint32_t decoder_attn_heads{16};
    const uint32_t decoder_query_heads{4};
    const uint32_t head_size{128};
    const uint32_t eos_token_id{1024};
    const uint32_t pad_token_id{1025};
    const uint32_t bos_token_id{1026};
    const uint32_t output_vocab_size{1028};
    const uint32_t audio_vocab_size{1024};
    const uint32_t max_generation_size{3072};
    const uint32_t max_encoder_context_length{1024};

    const float cfg_scale_data[2]{3.0, 1024.0};
    const uint32_t max_delay{15};
};

struct dia_model : tts_model, dia_model_constants {
    dia_encoder encoder;
    dia_decoder decoder;

    void assign_weight(sv name, ggml_tensor *tensor) override;
    explicit dia_model(shared_ptr<gpu_context> gpu, gguf_context * meta, ggml_context * weights)
        : tts_model{move(gpu), model_tensor_meta{weights, "dia", 130}}, dia_model_constants{meta}, encoder{n_encoder_layers}, decoder{n_decoder_layers, n_output_heads} {
    }
};

struct dia_context : runner_context {
    explicit dia_context(dia_model * model, int n_threads, bool use_cpu = true);

    uint32_t current_position = 0;  // current position in the active sequence
    int delay_steps           = -1; // the max remaining steps to take before terminating; is set after an eos token is seen on the first output channel
    size_t prompt_size        = 0;

    uint32_t max_generation_size; // this is set by the generation context or defaults to the config set on dia model.

    vector<uint32_t> output_tokens;
    dia_model * model;

    ggml_tensor * inp_tokens;
    ggml_tensor * audio_inp_tokens;
    ggml_tensor * positions;
    ggml_tensor * encode_positions;
    ggml_tensor * encode_attn_mask;
    ggml_tensor * cross_attn_mask;

    void reset();
};

struct dia_kv_cache_layer {
    explicit dia_kv_cache_layer(ggml_context *ctx, int64_t gen_ne0, int64_t cross_ne0, int index);

    ggml_tensor * k;
    ggml_tensor * v;
    ggml_tensor * cross_k;
    ggml_tensor * cross_v;
};

struct dia_kv_cache {
    vector<dia_kv_cache_layer> layers;
    const ggml_context_ptr ctx;
    ggml_backend_buffer_type_t buft;
    ggml_backend_buffer_ptr buf;
};

struct dia_ubatch {
    size_t sequence_length; // for just audio tokens the sequence length should be the total_tokens / num_heads; for normal generation this should always be 1.
    bool encoder_step{}; // whether we are performing the prompt encoding in this step.
    size_t sentence_length; // the number of non padded tokens in the conditional context
    vector<uint32_t> tokens; // character tokens for the encoder
    vector<uint32_t> audio_tokens; // audio tokens from the last generation
};

// This struct is intended to support end-to-end TTS generation for the Dia model. As such, it manages Dia's model compilation, compute, generation,
// tokenizationm and sampling process, and uses the dac_runner struct to encode audio outputs.
struct dia_runner : tts_runner_with_dac {
    using model_type = dia_model;
    dia_runner(shared_ptr<dia_model> model): model{model} {
        decode_sampler.vocab_size = model->output_vocab_size;
    };
    shared_ptr<dia_model> model;
    dia_kv_cache kv_cross_self{};
    sampler decode_sampler{};

    void tokenize_sentence(str sentence, dia_ubatch & tokens);
    dia_ubatch batch_from_sentence(str sentence);
    void configure_generation(const generation_configuration & config) override;
    void assign_weight(sv name, ggml_tensor *tensor) override;
    dia_ubatch build_worst_case_batch();
    ggml_cgraph * build_dia_graph(dia_ubatch & batch);
    void set_inputs(dia_ubatch & batch);
    void decode(dia_ubatch & batch);
    void prepare_post_load();
    int generate(str sentence, struct tts_response * response) override;
    bool check_stopping(dia_ubatch & batch);
    void adjust_output_tokens(vector<uint32_t> & output_tokens, vector<uint32_t> & filtered);
    int generate_from_batch(dia_ubatch & batch, struct tts_response * output);
};
