#include "dac_runner.h"
#include "ggml-backend.h"

void dac_context::set_threads() {
    if (backend != nullptr) {
#ifdef GGML_METAL
        // this is form copied from llama.cpp, but has since been removed. I don't know if this should be tuned.
        ggml_backend_metal_set_n_cb(backend, 2);
#endif
    }
    if (backend_cpu != nullptr) {
        ggml_backend_cpu_set_n_threads(backend_cpu, n_threads);
        ggml_backend_cpu_set_threadpool(backend_cpu, threadpool);
    }
}

void dac_context::build_schedule() {
    backend_cpu_buffer = ggml_backend_cpu_buffer_type();
    if (backend != nullptr) {
#ifdef GGML_METAL
        backend_buffer = ggml_backend_metal_buffer_type();
#endif
        std::vector<ggml_backend_buffer_type_t> bufs = {backend_buffer, backend_cpu_buffer};
        std::vector<ggml_backend_t> backs = {backend, backend_cpu};
        sched = ggml_backend_sched_new(backs.data(), bufs.data(), 2, model->max_nodes(), false);
    } else {
        std::vector<ggml_backend_buffer_type_t> bufs = {backend_cpu_buffer};
        std::vector<ggml_backend_t> backs = {backend_cpu};
        sched = ggml_backend_sched_new(backs.data(), bufs.data(), 1, model->max_nodes(), false);
    }
}

bool dac_context::prep_schedule(struct ggml_cgraph * gf) {
    return ggml_backend_sched_reserve(sched, gf);
}

static struct ggml_tensor * dac_build_audio_inputs(struct ggml_context * ctx, struct dac_context * dctx, const dac_ubatch & batch, std::vector<dac_quantize_layer> layers) {
    struct ggml_tensor * embd;
    
    dctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.sequence_length*dctx->model->n_heads);
    ggml_set_input(dctx->inp_tokens);
    if (dctx->backend) {
        ggml_backend_sched_set_tensor_backend(dctx->sched, dctx->inp_tokens, dctx->backend);
    }

    for(int i = 0; i < dctx->model->n_heads; i++) {
        auto quantize_layer = dctx->model->quantizer_layers[i];
        struct ggml_tensor * code = ggml_cont(ctx, ggml_view_2d(ctx, dctx->inp_tokens, 1, batch.sequence_length, dctx->model->n_heads*ggml_type_size(GGML_TYPE_I32), i*ggml_type_size(GGML_TYPE_I32)));
        code = ggml_reshape_1d(ctx, code, batch.sequence_length);
        code = ggml_get_rows(ctx, quantize_layer.codebook, code);
        code = ggml_cont(ctx, ggml_transpose(ctx, code));
        code = ggml_conv_1d(ctx, quantize_layer.out_proj_kernel, code, 1, 0, 1);
        code = ggml_add(ctx, code, quantize_layer.out_proj_bias);

        if (i == 0) {
            embd = code;
        } else {
            embd = ggml_add(ctx, embd, code);
        }
    }
    return embd;
}

static struct ggml_tensor * build_residual_unit(ggml_context * ctx, struct ggml_tensor * cur, dac_residual_unit & u, int padding, int dilation) {
    struct ggml_tensor * residual = cur;
    cur = dac_snake_1d(ctx, u.in_snake_alpha, cur);
    cur = ggml_conv_1d(ctx, u.in_conv_kernel, cur, 1, padding, dilation);
    cur = ggml_add(ctx, cur, u.in_conv_bias);
    cur = dac_snake_1d(ctx, u.out_snake_alpha, cur);
    cur = ggml_conv_1d(ctx, u.out_conv_kernel,  cur, 1, 0, 1);
    cur = ggml_add(ctx, cur, u.out_conv_bias);
    return ggml_add(ctx, cur, residual);
}

static struct ggml_tensor * build_decoder_block(ggml_context * ctx, struct ggml_tensor * cur, dac_layer & l, struct dac_context * dctx) {
    cur = dac_snake_1d(ctx, l.snake_alpha_in, cur);
    cur = ggml_conv_transpose_1d(ctx, l.out_conv_kernel, cur, l.stride, l.padding, 1);
    cur = ggml_add(ctx, cur, l.out_conv_bias);
    for (int i = 0; i < l.residual_blocks.size(); i++) {
        cur = build_residual_unit(ctx, cur, l.residual_blocks[i], pow(3, (i + 1)), pow(3, i));
    }
    return cur;
}

struct dac_context * build_new_dac_context(struct dac_model * model, int n_threads, bool use_cpu) {
    dac_context * dctx = new dac_context(model, n_threads);
    if (!use_cpu) {
#ifdef GGML_METAL
        dctx->backend = ggml_backend_metal_init();
#endif
    }
    dctx->backend_cpu = ggml_backend_cpu_init();
    dctx->set_threads();
    dctx->build_schedule();
    dctx->buf_compute_meta.resize(ggml_tensor_overhead()*model->max_nodes() + ggml_graph_overhead_custom(model->max_nodes(), false));
    return dctx;
}


void dac_runner::init_build() {
    struct ggml_init_params params = {
        /*.mem_size   =*/ dctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ dctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx = ggml_init(params);
}
    
void dac_runner::free_build() {
    if (ctx) {
        ggml_free(ctx);
        ctx = nullptr;
    }
}

void dac_runner::prepare_post_load() {
    dac_ubatch batch;
    batch.sequence_length = model->max_generation_size;
    ggml_cgraph * gf = build_dac_graph(batch);
    dctx->prep_schedule(gf);
}
    
struct ggml_cgraph * dac_runner::build_dac_graph(dac_ubatch & batch) {
    init_build();
    // splitting this out from the primary graph so that we can better manage streaming (i.e. sentence chunks are better performed this way)
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    
    struct ggml_tensor * cur;
    struct ggml_tensor * inputs;
    
    inputs = dac_build_audio_inputs(ctx, dctx, batch, model->quantizer_layers);
    ggml_set_name(inputs, "quanitzed_inputs");
    
    // everything besides the inputs is just a forward pass
    cur = ggml_conv_1d(ctx, model->in_conv_kernel, inputs, 1, 3, 1);
    cur = ggml_add(ctx, cur, model->in_conv_bias);
    for (auto l : model->layers) {
        cur = build_decoder_block(ctx, cur, l, dctx);
    }
    cur = dac_snake_1d(ctx, model->snake_alpha, cur);
    cur = ggml_conv_1d(ctx, model->out_conv_kernel, cur, 1, 3, 1);
    cur = ggml_add(ctx, cur, model->out_conv_bias);
    cur = ggml_tanh(ctx, cur);
    ggml_build_forward_expand(gf, cur);
    free_build();
    return gf;
}

void dac_runner::run(uint32_t * input_tokens, uint32_t sequence_length, struct tts_response * outputs) {
    dac_ubatch batch;
    batch.input_tokens = input_tokens;
    batch.sequence_length = sequence_length;
    ggml_backend_sched_reset(dctx->sched);
    
    const size_t prev_size = dctx->buf_output ? ggml_backend_buffer_get_size(dctx->buf_output) : 0;
    const size_t new_size = model->max_generation_size * model->up_sampling_factor * sizeof(float);
    
    if (!dctx->buf_output || prev_size < new_size) {
        if (dctx->buf_output) {
            ggml_backend_buffer_free(dctx->buf_output);
            dctx->buf_output = nullptr;
            dctx->logits = nullptr;
        }

        dctx->buf_output = ggml_backend_buft_alloc_buffer(dctx->backend_cpu_buffer, new_size);
    }
    
    outputs->data = (float *) ggml_backend_buffer_get_base(dctx->buf_output);
    ggml_backend_buffer_clear(dctx->buf_output, 0);
    
    struct ggml_cgraph * gf = NULL;
    gf = build_dac_graph(batch);
    
    // the output is always the last tensor in the graph
    struct ggml_tensor * result = gf->nodes[gf->n_nodes - 1];
    ggml_backend_sched_alloc_graph(dctx->sched, gf);
    
    ggml_backend_tensor_set(dctx->inp_tokens, batch.input_tokens, 0, batch.sequence_length*model->n_heads*ggml_element_size(dctx->inp_tokens));

    ggml_backend_sched_graph_compute_async(dctx->sched, gf);

    ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(dctx->sched, result);

    ggml_backend_tensor_get_async(backend_res, result, outputs->data, 0, batch.sequence_length*sizeof(float)*model->up_sampling_factor);

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(dctx->sched);
    outputs->n_outputs = sequence_length * model->up_sampling_factor;
    return;
}
