#pragma once

#include <stdlib.h>
#include "tts_model.h"
#include "tokenizer.h"
#include "phonemizer.h"

// Rather than using ISO 639-2 language codes, Kokoro voice pack specify their corresponding language via their first letter.
// Below is a map that describes the relationship between those designations and espeak-ng's voice identifiers so that the 
// appropriate phonemization protocol can inferred from the Kokoro voice.
constexpr array<str, 255> KOKORO_LANG_TO_ESPEAK_ID = []{
    array<str, 255> result{};
    result['a'] = "gmw/en-US";
    result['b'] = "gmw/en";
    result['e'] = "roa/es";
    result['f'] = "roa/fr";
    result['h'] = "inc/hi";
    result['i'] = "roa/it";
    result['j'] = "jpx/ja";
    result['p'] = "roa/pt-BR";
    result['z'] = "sit/cmn";
    return result;
}();

struct lstm_cell {
    vector<ggml_tensor *> weights;
    vector<ggml_tensor *> biases;
    vector<ggml_tensor *> reverse_weights;
    vector<ggml_tensor *> reverse_biases;
};

struct lstm {
    vector<ggml_tensor *> hidden;
    vector<ggml_tensor *> states;

    bool bidirectional{};
    vector<lstm_cell> cells;
};

struct duration_predictor_layer {
    lstm * rnn;
    ggml_tensor * ada_norm_gamma_weight;
    ggml_tensor * ada_norm_gamma_bias;
    ggml_tensor * ada_norm_beta_weight;
    ggml_tensor * ada_norm_beta_bias;
};

struct ada_residual_conv_block {
    ggml_tensor * conv1;
    ggml_tensor * conv1_bias;
    ggml_tensor * conv2;
    ggml_tensor * conv2_bias;
    // TODO merge gamma and beta for matmul, then use views to unmerge
    ggml_tensor * norm1_gamma;
    ggml_tensor * norm1_gamma_bias;
    ggml_tensor * norm1_beta;
    ggml_tensor * norm1_beta_bias;
    ggml_tensor * norm2_gamma;
    ggml_tensor * norm2_gamma_bias;
    ggml_tensor * norm2_beta;
    ggml_tensor * norm2_beta_bias;
    ggml_tensor * pool = nullptr;
    ggml_tensor * pool_bias = nullptr;
    ggml_tensor * upsample = nullptr;
    ggml_tensor * upsample_bias = nullptr;
};

struct duration_predictor {
    ggml_tensor * albert_encode;
    ggml_tensor * albert_encode_bias;
    vector<duration_predictor_layer *> layers;
    lstm * duration_proj_lstm;
    ggml_tensor * duration_proj;
    ggml_tensor * duration_proj_bias;
    ggml_tensor * n_proj_kernel;
    ggml_tensor * n_proj_bias;
    ggml_tensor * f0_proj_kernel;
    ggml_tensor * f0_proj_bias;
    lstm * shared_lstm;
    vector<ada_residual_conv_block*> f0_blocks;
    vector<ada_residual_conv_block*> n_blocks;
};

struct kokoro_text_encoder_conv_layer {
    ggml_tensor * norm_gamma;
    ggml_tensor * norm_beta;
    ggml_tensor * conv_weight;
    ggml_tensor * conv_bias;
};

struct kokoro_text_encoder {
    ggml_tensor * embd;
    vector<kokoro_text_encoder_conv_layer *> conv_layers;
    lstm * out_lstm;
};

struct kokoro_generator_residual_block {
    vector<uint32_t> conv1_dilations;
    vector<uint32_t> conv1_paddings;

    vector<ggml_tensor *> adain1d_1_gamma_weights;
    vector<ggml_tensor *> adain1d_2_gamma_weights;
    vector<ggml_tensor *> adain1d_1_gamma_biases;
    vector<ggml_tensor *> adain1d_2_gamma_biases;
    vector<ggml_tensor *> adain1d_1_beta_weights;
    vector<ggml_tensor *> adain1d_2_beta_weights;
    vector<ggml_tensor *> adain1d_1_beta_biases;
    vector<ggml_tensor *> adain1d_2_beta_biases;
    vector<ggml_tensor *> input_alphas;
    vector<ggml_tensor *> output_alphas;
    vector<ggml_tensor *> convs1_weights;
    vector<ggml_tensor *> convs1_biases;
    vector<ggml_tensor *> convs2_weights;
    vector<ggml_tensor *> convs2_biases;
};

struct kokoro_noise_residual_block {
    uint32_t input_conv_stride;
    uint32_t input_conv_padding;

    ggml_tensor * input_conv;
    ggml_tensor * input_conv_bias;
    struct kokoro_generator_residual_block * res_block;
};

struct kokoro_generator_upsample_block {
    uint32_t padding;
    uint32_t stride;

    // these are just conv transpose layers
    ggml_tensor * upsample_weight;
    ggml_tensor * upsample_bias;
};

struct kokoro_generator {
    // unfortunately the squared sum of the windows needs to be computed dynamically per run because it is dependent
    // on the sequence size of the generation and the hop is typically less than half the size of our window.
    ggml_tensor * window;

    ggml_tensor * m_source_weight;
    ggml_tensor * m_source_bias;
    ggml_tensor * out_conv_weight;
    ggml_tensor * out_conv_bias;
    vector<kokoro_noise_residual_block*> noise_blocks;
    vector<kokoro_generator_residual_block*> res_blocks;
    vector<kokoro_generator_upsample_block*> ups;
};

struct kokoro_decoder {
    ggml_tensor * f0_conv;
    ggml_tensor * f0_conv_bias;
    ggml_tensor * n_conv;
    ggml_tensor * n_conv_bias;
    ggml_tensor * asr_conv;
    ggml_tensor * asr_conv_bias;
    vector<ada_residual_conv_block*> decoder_blocks;
    ada_residual_conv_block* encoder_block;
    kokoro_generator * generator;
};

struct albert_layer {
    ggml_tensor * ffn;
    ggml_tensor * ffn_out;
    ggml_tensor * ffn_bias;
    ggml_tensor * ffn_out_bias;
    ggml_tensor * layer_output_norm_weight;
    ggml_tensor * layer_output_norm_bias;
    ggml_tensor * q;
    ggml_tensor * k;
    ggml_tensor * v;
    ggml_tensor * o;
    ggml_tensor * q_bias;
    ggml_tensor * k_bias;
    ggml_tensor * v_bias;
    ggml_tensor * o_bias;
    ggml_tensor * attn_norm_weight;
    ggml_tensor * attn_norm_bias;
};

struct kokoro_model : tts_model {
    // standard configruation for Kokoro's Albert model
    // tokenization
    uint32_t bos_token_id = 0;
    uint32_t eos_token_id = 0;
    uint32_t space_token_id = 16;
    // duration prediction
    uint32_t max_context_length = 512;
    uint32_t vocab_size = 178;
    uint32_t hidden_size = 768;
    uint32_t n_attn_heads = 12;
    uint32_t n_layers = 1;
    uint32_t n_recurrence = 12;
    uint32_t head_size = 64;
    uint32_t duration_hidden_size = 512;
    uint32_t up_sampling_factor;
    float upsample_scale = 300.0f;
    float scale = 0.125f;

    // standard configuration for duration prediction
    uint32_t f0_n_blocks = 3;
    uint32_t n_duration_prediction_layers = 3;
    // while it is technically possible for the duration predictor to assign 50 values per token there is no practical need to 
    // allocate that many items to the sequence as it is impossible for all tokens to require such long durations and each 
    // allocation increases node allocation size by O(N)
    uint32_t max_duration_per_token = 20;
    uint32_t style_half_size = 128;

    // standard text encoding configuration
    uint32_t n_conv_layers = 3;

    // standard decoder configuration
    uint32_t n_kernels = 3;
    uint32_t n_upsamples = 2;
    uint32_t n_decoder_blocks = 4;
    uint32_t n_res_blocks = 6;
    uint32_t n_noise_blocks = 2;
    uint32_t out_conv_padding = 3;
    uint32_t post_n_fft = 11;
    uint32_t true_n_fft = 20;
    uint32_t stft_hop = 5;
    uint32_t harmonic_num = 8;
    float sin_amp = 0.1f;
    float noise_std = 0.003f;
    float voice_threshold = 10.0f;
    float sample_rate = 24000.0f;
    string window = "hann";

    // It is really annoying that ggml doesn't allow using non ggml tensors as the operator for simple math ops.
    // This is just the constant defined above as a tensor.
    ggml_tensor * n_kernels_tensor;

    // Kokoro loads albert with use_pooling = true but doesn't use the pooling outputs.
    bool uses_pooling = false;
    bool static_token_types = true;

    map<string, ggml_tensor *> voices;

    // Albert portion of the model
    ggml_tensor * embd_hidden;
    ggml_tensor * embd_hidden_bias;
    ggml_tensor * token_type_embd = nullptr;
    ggml_tensor * token_embd;
    ggml_tensor * position_embd;
    ggml_tensor * input_norm_weight;
    ggml_tensor * input_norm_bias;
    ggml_tensor * static_token_type_values = nullptr;
    ggml_tensor * pool = nullptr;
    ggml_tensor * pool_bias = nullptr;
    vector<albert_layer *> layers;

    ggml_tensor * harmonic_sampling_norm = nullptr; // a static 1x9 harmonic multiplier
    ggml_tensor * sampling_factor_scalar = nullptr; // a static scalar
    ggml_tensor * sqrt_tensor = nullptr; // static tensor for constant division

    // Prosody Predictor portion of the model
    struct duration_predictor * prosody_pred;

    // Text encoding portion of the model
    struct kokoro_text_encoder * text_encoder;

    // Decoding and Generation portion of the model
    struct kokoro_decoder * decoder;

    // the default hidden states need to be initialized 
    vector<lstm*> lstms;

    size_t duration_node_counter = 0;
    size_t generation_node_counter = 0;
    // setting this is likely unnecessary as it is precomputed by the post load function.
    uint32_t post_load_tensor_bytes = 13000;

    size_t max_gen_nodes();
    size_t max_duration_nodes();

    lstm * prep_lstm();
    // helper functions for assigning tensors to substructs
    void assign_lstm(lstm * rnn, string name, ggml_tensor * tensor);
    void assign_generator_weight(kokoro_generator * generator, string name, ggml_tensor * tensor);
    void assign_gen_resblock(kokoro_generator_residual_block * block, string name, ggml_tensor * tensor);
    void assign_ada_res_block(ada_residual_conv_block * block, string name, ggml_tensor * tensor);
    void assign_decoder_weight(string name, ggml_tensor * tensor);
    void assign_duration_weight(string name, ggml_tensor * tensor);
    void assign_text_encoder_weight(string name, ggml_tensor * tensor);
    void assign_albert_weight(string name, ggml_tensor * tensor);


    void post_load_assign();
    void assign_weight(string name, ggml_tensor * tensor);
    void prep_layers(gguf_context * meta);
    void prep_constants(gguf_context * meta);
    explicit kokoro_model(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only = true) {
        function<void (ggml_tensor *)> fn = ([&](ggml_tensor * cur) {
            string name = ggml_get_name(cur);
            size_t increment = 1;
            if (name.find("lstm") != string::npos) {
                increment = max_context_length;
            }
            if (name.find("duration_predictor") != string::npos) {
                duration_node_counter += increment;
            } else {
                generation_node_counter += increment;
            }
        });
        compute_tensor_meta_cb = &fn;
        prep_constants(meta_ctx);
        prep_layers(meta_ctx);
        tts_model::setup_from_file(meta_ctx, load_context, cpu_only, "kokoro", 1.6, post_load_tensor_bytes);
    }
};

struct kokoro_ubatch {
    size_t n_tokens; // the number of tokens in our encoded sequence
    uint32_t * input_tokens;    // [n_tokens]
    struct kokoro_duration_response * resp = nullptr;
};

struct kokoro_duration_context : runner_context {
    kokoro_duration_context(kokoro_model * model, int n_threads): runner_context{n_threads, model->max_duration_nodes()*5}, model(model) {};
    ~kokoro_duration_context() {
        ggml_backend_buffer_free(buf_len_output);
    }
    
    string voice = "af_alloy";
    kokoro_model * model;
    ggml_backend_buffer_t buf_len_output = nullptr;

    
    size_t  logits_size = 0; // capacity (of floats) for logits
    float * logits      = nullptr;
    float * lens         = nullptr;
    
    ggml_tensor * inp_tokens;
    ggml_tensor * positions;
    ggml_tensor * attn_mask;
    ggml_tensor * token_types = nullptr;
};

static ggml_tensor * build_albert_attn_mask(ggml_context * ctx, struct kokoro_duration_context *kctx, const kokoro_ubatch & batch);
static ggml_tensor * build_albert_inputs(ggml_context * ctx, kokoro_model * model, ggml_tensor * input_tokens, ggml_tensor * positions, ggml_tensor * token_types);
static ggml_tensor * build_albert_norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * weight, ggml_tensor * bias);
static ggml_tensor * build_lstm(ggml_context * ctx, ggml_tensor * input, lstm* rnn, uint32_t sequence_length);
static ggml_tensor * build_lstm_run(ggml_context * ctx, ggml_tensor * input, ggml_tensor * h_0, ggml_tensor * c_0, vector<ggml_tensor *> weights, vector<ggml_tensor *> biases, uint32_t sequence_length, bool reversed = false);
static ggml_tensor * build_ada_residual_conv(ggml_context * ctx, ggml_tensor * x, ada_residual_conv_block * block, ggml_tensor * style, ggml_tensor * sqrt_tensor);
static ggml_tensor * build_kokoro_generator_res_block(ggml_context * ctx, ggml_tensor * x, ggml_tensor * style, kokoro_generator_residual_block * block);
static ggml_tensor * build_noise_block(ggml_context * ctx, kokoro_noise_residual_block * block, ggml_tensor * x, ggml_tensor * style);
static kokoro_generator_residual_block * build_res_block_from_file(gguf_context * meta, string base_config_key);
static kokoro_noise_residual_block * build_noise_block_from_file(gguf_context * meta, int index);
static kokoro_generator_upsample_block* kokoro_generator_upsample_block(gguf_context * meta, int index);

string get_espeak_id_from_kokoro_voice(string voice);
struct kokoro_duration_context * build_new_duration_kokoro_context(struct kokoro_model * model, int n_threads, bool use_cpu = true);

struct kokoro_duration_response {
    size_t n_outputs;
    float * lengths;
    float * hidden_states;
};

// This struct is intended to manage graph and compute for the duration prediction portion of the kokoro model.
// Duration computation and speech generation are separated into distinct graphs because the precomputed graph structure of ggml doesn't 
// support the tensor dependent views that would otherwise be necessary.
struct kokoro_duration_runner {
    kokoro_duration_runner(shared_ptr<kokoro_model> model, kokoro_duration_context * context, single_pass_tokenizer * tokenizer): model{model}, kctx(context), tokenizer(tokenizer) {};
    single_pass_tokenizer * tokenizer;
    shared_ptr<kokoro_model> model;
    kokoro_duration_context * kctx;

    void prepare_post_load();
    struct kokoro_ubatch build_worst_case_batch();
    void set_inputs(kokoro_ubatch & batch);
    struct ggml_cgraph * build_kokoro_duration_graph(kokoro_ubatch & batch);
    void run(kokoro_ubatch & ubatch);
};

struct kokoro_context : runner_context {
    kokoro_context(kokoro_model * model, int n_threads): runner_context(n_threads, model->max_gen_nodes()*30), model(model) {};

    string voice = "af_alloy";
    
    kokoro_model * model;

    uint32_t total_duration;
    uint32_t sequence_length;
    
    size_t  logits_size = 0; // capacity (of floats) for logits
    float * logits      = nullptr;
    
    ggml_tensor * inp_tokens;
    ggml_tensor * duration_pred;
    ggml_tensor * duration_mask;
    ggml_tensor * window_sq_sum; // needs to be calculatd from the generator window.
    ggml_tensor * uv_noise_data;
};

// TODO: now that we are passing the context down to these methods we should clean up their parameters
static ggml_tensor * build_generator(ggml_context * ctx, kokoro_model * model, kokoro_context * kctx, ggml_tensor * x, ggml_tensor * style, ggml_tensor * f0_curve, kokoro_generator * generator, int sequence_length, ggml_tensor * window_sq_sum, ggml_cgraph * gf);
static ggml_tensor * build_sin_gen(ggml_context * ctx, kokoro_model * model, kokoro_context * kctx, ggml_tensor * x, int harmonic_num, int sequence_length, float voice_threshold, float sin_amp, float noise_std);

struct kokoro_context * build_new_kokoro_context(struct kokoro_model * model, int n_threads, bool use_cpu = true);

// This manages the graph compilation of computation for the Kokoro model.
struct kokoro_runner : tts_runner {
    kokoro_runner(shared_ptr<kokoro_model> model, gguf_context * meta_ctx):
    model{model}, tokenizer{meta_ctx, "tokenizer.ggml.tokens"}, pmzr{meta_ctx, config->espeak_voice_id} {
        tts_runner::sampling_rate = 24000.0f;
    }
    const shared_ptr<kokoro_model> model;
    single_pass_tokenizer tokenizer;
    kokoro_context kctx;
    kokoro_duration_runner drunner;
    phonemizer phmzr{meta_ctx, espeak_voice_id};
    str voice;
    str voice_code{""};

    string default_voice = "af_alloy";

    vector<vector<uint32_t>> tokenize_chunks(vector<string> clauses);
    void assign_weight(sv name, ggml_tensor *tensor) override;
    void prepare_post_load();
    kokoro_ubatch build_worst_case_batch();
    void set_inputs(kokoro_ubatch & batch, uint32_t total_size);
    struct ggml_cgraph * build_kokoro_graph(kokoro_ubatch & batch);
    void run(kokoro_ubatch & batch, struct tts_response * outputs);
    int generate(str prompt, tts_response * response) override;
    void configure_generation(const generation_configuration & config) override {
        voice = config.voice;
        voice_code = config.espeak_voice_id;
    }
};
