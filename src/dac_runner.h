#ifndef dac_runner_h
#define dac_runner_h

#include "parler_gguf.h"

struct dac_context {
    dac_context(dac_model * model, int n_threads): model(model), n_threads(n_threads) {};
    ~dac_context() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend_cpu);
        if (backend) {
            ggml_backend_free(backend);
        }
        if (buf_output) {
            ggml_backend_buffer_free(buf_output);
        }
    }
    
    struct dac_model * model;
    
    // TODO: extend the backend and buffer support out to all devices
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t backend_buffer = nullptr;

    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_type_t backend_cpu_buffer = nullptr;
    
    std::vector<uint8_t> buf_compute_meta;
    ggml_backend_buffer_t buf_output = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_threadpool_t threadpool = nullptr;
    int n_threads;
    
    size_t  logits_size = 0; // capacity (of floats) for logits
    float * logits      = nullptr;
    
    struct ggml_tensor * inp_tokens;
    
    void set_threads();
    void build_schedule();
    bool prep_schedule(ggml_cgraph * gf);
};

struct dac_context * build_new_dac_context(struct dac_model * model, int n_threads, bool use_cpu = true);

struct dac_ubatch {
    uint32_t * input_tokens;
    uint32_t sequence_length;
};

static struct ggml_tensor * dac_build_audio_inputs(struct ggml_context * ctx, struct dac_context * dctx, const dac_ubatch & batch, std::vector<dac_quantize_layer> layers);
static struct ggml_tensor * build_residual_unit(ggml_context * ctx, struct ggml_tensor * cur, dac_residual_unit & u, int padding, int dilation);
static struct ggml_tensor * build_decoder_block(ggml_context * ctx, struct ggml_tensor * cur, dac_layer & l);

// This struct is intended to manage the dac model's graph compilation and compute function.
struct dac_runner {
    dac_runner(dac_model * model, dac_context * context): model(model), dctx(context) {};
    ~dac_runner() {
        if (ctx) {
            ggml_free(ctx);
        }
        model->free();
        delete model;
        delete dctx;
    }
    dac_model * model;
    dac_context * dctx;
    struct ggml_context * ctx = nullptr;
    
    void init_build();
    void free_build();
    
    void prepare_post_load();
    struct ggml_cgraph * build_dac_graph(dac_ubatch & batch);
    void run(uint32_t * input_tokens, uint32_t sequence_length, std::vector<float> * outputs);
};

#endif
