#ifndef parler_h
#define parler_h

#include "dac_runner.h"
#include "sampler.h"
#include <thread>
#include <fstream>

struct parler_context {
    parler_context(parler_tts_model * model, int n_threads): model(model), n_threads(n_threads) {};
    ~parler_context() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend_cpu);
        if (backend) {
            ggml_backend_free(backend);
        }
        if (buf_output) {
            ggml_backend_buffer_free(buf_output);
        }
    }
    
    struct parler_tts_model * model;
    
    // TODO: extend the backend and buffer support out to all devices
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t backend_buffer = nullptr;
    std::vector<bool> eos_seen;
    
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_type_t backend_cpu_buffer = nullptr;

    std::vector<uint8_t> buf_compute_meta;
    ggml_backend_buffer_t buf_output = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_threadpool_t threadpool = nullptr;
    int n_threads;
    bool use_cache = true;
    
    size_t  output_size = 0; // capacity (of tokens positions) for the output buffers
    int32_t n_outputs   = 0; // number of actually-used outputs in the current ubatch or last logical batch
    uint32_t current_position = 0; // current position in the active sequence
    uint32_t prompt_end_position = 0; // the position of the text prompt termination (used for adjusting the cache when incrementally generating)
    int32_t seq_id; // a unique identifier associated with the active sequence.
    
    std::vector<uint32_t> output_tokens;

    size_t  logits_size = 0; // capacity (of floats) for logits
    float * logits      = nullptr;
    
    struct ggml_tensor * inp_tokens;
    struct ggml_tensor * audio_inp_tokens;
    struct ggml_tensor * positions;
    struct ggml_tensor * attn_mask;
    struct ggml_tensor * attn_mask_cross;
    
    void reset(int32_t n_output_heads);
    void set_threads();
    void build_schedule();
    bool prep_schedule(ggml_cgraph * gf);
};

struct parler_kv_cache {
    int32_t seq_id;
    
    ggml_type type_k = GGML_TYPE_F32;
    ggml_type type_v = GGML_TYPE_F32;

    std::vector<struct ggml_tensor *> k_l;
    std::vector<struct ggml_tensor *> v_l;
    
    struct ggml_context * ctx;
    ggml_backend_buffer_type_t buft;
    ggml_backend_buffer_t buf;
    
    void free() {
        ggml_free(ctx);
        ggml_backend_buffer_free(buf);
    }

    ~parler_kv_cache() {
        free();
    }
};

struct parler_ubatch {
    parler_ubatch(bool audio_generation, size_t n_tokens, size_t n_audio_tokens, size_t sequence_length, 
        uint32_t * tokens, uint32_t * audio_tokens, uint32_t * positions, uint32_t * true_order, 
        int current_step): audio_generation(audio_generation), n_tokens(n_tokens), n_audio_tokens(n_audio_tokens), sequence_length(sequence_length), tokens(tokens), audio_tokens(audio_tokens), positions(positions), true_order(true_order), current_step(current_step) {};
    parler_ubatch() {};
    bool audio_generation; // whether we are receiving codebook decoded tokens or text tokens
    size_t n_tokens; // total sentence tokens
    size_t n_audio_tokens; // total audio tokens
    size_t sequence_length; // for just audio tokens the sequence length should be the total_tokens / num_heads; in general this should be n_tokens + n_audio_tokens / num_heads
    uint32_t * tokens;    // [n_tokens]
    uint32_t * audio_tokens; // [n_audio_tokens]
    uint32_t * positions; // [sequence_length]
    uint32_t * true_order;
    int current_step = 0; // total_generations
};

struct parler_context * build_new_parler_context(struct parler_tts_model * model, int n_threads, bool use_cpu = true);
static bool parler_kv_cache_init(struct parler_kv_cache * cache, parler_tts_model * model, parler_context * pctx, int32_t seq_id);

struct ggml_tensor * parler_build_inp_embd(struct ggml_context * ctx, struct parler_context * pctx, parler_tts_model * model, const parler_ubatch & batch);
struct ggml_tensor * parler_build_layer_norm(struct ggml_context * ctx, struct ggml_tensor * inputs, struct ggml_tensor * weight, struct ggml_tensor * bias);
void parler_build_kv_store(struct ggml_context * ctx, const parler_kv_cache * kv, struct ggml_cgraph * graph, struct ggml_tensor * k_cur, struct ggml_tensor * v_cur, int32_t n_tokens, int32_t kv_head, int32_t index, int32_t n_embd_gqa);
struct ggml_tensor * parler_build_head_outputs(struct ggml_context * ctx, parler_tts_model * model, struct ggml_tensor * cur);
struct ggml_tensor * build_attn_mask(ggml_context * ctx, parler_context * pctx, parler_ubatch & batch);
struct ggml_tensor * build_attn_mask_cross(ggml_context * ctx, parler_context * pctx, parler_tts_model * model, parler_ubatch & batch);
static struct parler_ubatch batch_from_sentence(std::string sentence, parler_tts_model * model, unigram_tokenizer * tokenizer);

// This struct is intended to support end-to-end TTS generation. As such, it manages the parler tts model compilation, compute and generation process,
// the tokenization and sampling process, and uses the dac_runner struct to encode audio outputs.
struct parler_tts_runner {
    parler_tts_runner(parler_tts_model * model, dac_runner * audio_decoder, parler_context * pctx, unigram_tokenizer * ut, sampler * samp, parler_kv_cache * cache): model(model), dac_runner(audio_decoder), pctx(pctx), tokenizer(ut), sampler(samp), kv_self(cache) {};
    ~parler_tts_runner() {
        if (ctx) {
            ggml_free(ctx);
        }
        model->free();
        delete model;
        delete kv_self;
        delete dac_runner;
        delete pctx;
        delete sampler;
    }
    struct parler_tts_model * model;
    struct dac_runner * dac_runner;
    struct parler_context * pctx;
    struct unigram_tokenizer * tokenizer;
    struct parler_kv_cache * kv_self = nullptr;
    struct sampler * sampler;
    struct ggml_context * ctx = nullptr;

    void init_build();
    void free_build();
    parler_ubatch build_worst_case_batch();
    struct ggml_cgraph * build_parler_graph(parler_ubatch & batch);
    void set_inputs(parler_ubatch & batch);
    int decode(parler_ubatch & batch);
    void prepare_post_load();
    bool adjust_for_sequence_continuation(struct parler_ubatch & batch);
    int generate(std::string sentence, std::vector<float> * output, int32_t seq_id = -1);
    bool check_stopping();
    void adjust_output_tokens(std::vector<uint32_t> & output_tokens, std::vector<uint32_t> & filtered);
    int generate_from_batch(parler_ubatch & batch, std::vector<float> * output);
    void parler_graph_compute(ggml_cgraph * gf);
    void just_decode(uint32_t * tokens, int32_t sq_len, std::vector<float> * outputs);
    int generate_audio_tokens(std::string sentence);
};

struct parler_tts_runner * runner_from_file(const std::string & fname, int n_threads, bool cpu_only = true);

struct quantization_params {
    quantization_params(uint32_t n_threads, enum ggml_type quantize_type, void * imatrix = nullptr): n_threads(n_threads), quantize_type(quantize_type), imatrix(imatrix) {};
    uint32_t n_threads;
    enum ggml_type quantize_type; // quantization type
    void * imatrix = nullptr; // pointer to importance matrix data
    bool quantize_output_heads = false;
    bool quantize_text_embeddings = false;
    bool quantize_cross_attn_kv = false;
};

void quantize_gguf(const std::string & ifile, const std::string & ofile, struct quantization_params * params);

#endif
