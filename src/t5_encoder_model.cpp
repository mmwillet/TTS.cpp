#include "t5_encoder_model.h"

void t5_encoder::prep_layers(gguf_context * meta) {
	for (uint32_t i = 0; i < n_layers; i++) {
		t5_layer l;
		layers.push_back(l);
	}
}

void t5_encoder::prep_buffers_and_context(bool cpu_only) {
    if (cpu_only) {
        backend = ggml_backend_cpu_init();
        buffer = ggml_backend_cpu_buffer_type();
    } else {
#ifdef GGML_USE_METAL
        backend = ggml_backend_metal_init();
        buffer = ggml_backend_metal_buffer_type();
#endif
        // if use metal is not installed then we need to warn here
        if (!backend || !buffer) {
            TTS_ABORT("'GGML_USE_METAL' is not defined either set the model to use CPU only or install ggml with metal support.");
        }
    }
    size_t ctx_size = ggml_tensor_overhead() * (size_t) (tensor_meta.n_tensors * 1.25);
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ctx = ggml_init(params);
    buf = ggml_backend_buft_alloc_buffer(buffer, tensor_meta.n_bytes);
    return;
}

void t5_encoder::prep_constants(gguf_context * meta) {
	int n_layers_key = gguf_find_key(meta, "t5encoder.block_count");
    if (n_layers_key != -1) {
        n_layers = gguf_get_val_u32(meta, n_layers_key);
    }

	int hidden_size_key = gguf_find_key(meta, "t5encoder.embedding_length");
    if (hidden_size_key != -1) {
        hidden_size = gguf_get_val_u32(meta, hidden_size_key);
    }

	int attn_heads_key = gguf_find_key(meta, "t5encoder.attention.head_count");
    if (attn_heads_key != -1) {
        n_attn_heads = gguf_get_val_u32(meta, attn_heads_key);
    }

    int context_size_key = gguf_find_key(meta, "t5encoder.context_length");
    if (context_size_key != -1) {
        max_context_length = gguf_get_val_u32(meta, context_size_key);
    }

    int bos_token_id_key = gguf_find_key(meta, "tokenizer.ggml.bos_token_id");
    if (bos_token_id_key != -1) {
        bos_token_id = gguf_get_val_u32(meta, bos_token_id_key);
    }    

    int eos_token_id_key = gguf_find_key(meta, "tokenizer.ggml.eos_token_id");
    if (eos_token_id_key != -1) {
        eos_token_id = gguf_get_val_u32(meta, eos_token_id_key);
    }

    int vocab_size_key = gguf_find_key(meta, "t5encoder.vocab_size");
    if (vocab_size_key == -1) {
        TTS_ABORT("key 't5encoder.vocab_size' must be specified in gguf file.");
    }
    vocab_size = gguf_get_val_u32(meta, vocab_size_key);

    int output_size_key = gguf_find_key(meta, "t5encoder.output_size");
    if (output_size_key != -1) {
        output_size = gguf_get_val_u32(meta, output_size_key);
    }
}

void t5_encoder::setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only) {
    prep_constants(meta_ctx);
    prep_layers(meta_ctx);
    tensor_meta = compute_tensor_meta("t5encoder", load_context);
    prep_buffers_and_context(cpu_only);
}

void t5_encoder::set_tensor(struct ggml_tensor * tensor, struct ggml_tensor * target) {
	tensor->buffer = buf;
    tensor->data = (void *)((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
    size_t size = ggml_nbytes(target);
    ggml_backend_tensor_set(tensor, target->data, 0, size);
    ggml_set_name(tensor, target->name);
    offset += size;
}

size_t t5_encoder::max_nodes() {
    return std::max<size_t>(8192, tensor_meta.n_tensors*5);
}

void t5_encoder::free() {
    if (ctx) {
        ggml_free(ctx);
    }
    if (buf) {
        ggml_backend_buffer_free(buf);
    }
    if (backend) {
        ggml_backend_free(backend);
    }
}

void t5_context::set_threads() {
    if (backend != nullptr) {
#ifdef GGML_USE_METAL
        // this is form copied from llama.cpp, but has since been removed. I don't know if this should be tuned.
        ggml_backend_metal_set_n_cb(backend, 1);
#endif
    }
    if (backend_cpu != nullptr) {
        ggml_backend_cpu_set_n_threads(backend_cpu, n_threads);
        ggml_backend_cpu_set_threadpool(backend_cpu, threadpool);
    }
}

void t5_context::build_schedule() {
    backend_cpu_buffer = ggml_backend_cpu_buffer_type();
    if (backend != nullptr) {
#ifdef GGML_USE_METAL
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

bool t5_context::prep_schedule(ggml_cgraph * gf) {
    return ggml_backend_sched_reserve(sched, gf);
}

struct t5_context * build_new_t5_context(struct t5_encoder * model, int n_threads, bool use_cpu) {
	t5_context * t5ctx = new t5_context(model, n_threads);
    if (!use_cpu) {
#ifdef GGML_USE_METAL
        t5ctx->backend = ggml_backend_metal_init();
#endif
    }
    t5ctx->backend_cpu = ggml_backend_cpu_init();
    t5ctx->set_threads();
    t5ctx->build_schedule();
    t5ctx->buf_compute_meta.resize(ggml_tensor_overhead()*model->max_nodes() + ggml_graph_overhead_custom(model->max_nodes(), false));
    return t5ctx;
}

static struct ggml_tensor * build_t5_norm(struct ggml_context * ctx, struct ggml_tensor * cur, struct ggml_tensor * weight) {
	// this is static for all versions of t5 flan
    float eps = 0.000001;
    cur = ggml_rms_norm(ctx, cur, eps);
    cur = ggml_mul(ctx, cur, weight);
    return cur;
}

static struct ggml_tensor * build_t5_attn_mask(ggml_context * ctx, struct t5_context *t5ctx, const t5_ubatch & batch) {
    t5ctx->attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) batch.n_tokens, (int64_t) batch.n_tokens);
    ggml_set_input(t5ctx->attn_mask);

    return t5ctx->attn_mask;
}

static struct ggml_tensor * build_t5_pos_bias(ggml_context * ctx, struct ggml_tensor * pos_bucket, struct ggml_tensor * relative_attn_bias) {
    struct ggml_tensor * pos_bucket_1d = ggml_view_1d(ctx, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1], 0);
    struct ggml_tensor * pos_bias = ggml_get_rows(ctx, relative_attn_bias, pos_bucket_1d);

    pos_bias = ggml_view_3d(ctx, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1], ggml_element_size(pos_bias) * pos_bias->ne[0], ggml_element_size(pos_bias) * pos_bias->ne[0] * pos_bucket->ne[0],  0);
    pos_bias = ggml_permute(ctx, pos_bias, 2, 1, 0, 3);
    pos_bias = ggml_cont(ctx, pos_bias);
    return pos_bias;
}

void t5_runner::init_build() {
    struct ggml_init_params params = {
        /*.mem_size   =*/ t5ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ t5ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx = ggml_init(params);
}

void t5_runner::free_build() {
    if (ctx) {
        ggml_free(ctx);
        ctx = nullptr;
    }
}

t5_ubatch t5_runner::build_worst_case_batch()  {
    struct t5_ubatch batch;
    batch.n_tokens = model->max_context_length;
    return batch;
}

void t5_runner::prepare_post_load() {
    auto batch = build_worst_case_batch();
    auto gf = build_t5_graph(batch);
    t5ctx->prep_schedule(gf);
}

struct ggml_cgraph * t5_runner::build_t5_graph(t5_ubatch & batch) {
    init_build();
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;
    
    //t5ctx->positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    //ggml_set_input(t5ctx->positions);

    t5ctx->inp_pos_bucket = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, batch.n_tokens, batch.n_tokens);
    ggml_set_input(t5ctx->inp_pos_bucket);

    t5ctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    ggml_set_input(t5ctx->inp_tokens);

    inpL = ggml_get_rows(ctx, model->embd, t5ctx->inp_tokens);

    struct ggml_tensor * KQ_mask_dec = build_t5_attn_mask(ctx, t5ctx, batch);
    struct ggml_tensor * pos_bias = build_t5_pos_bias(ctx, t5ctx->inp_pos_bucket, model->relative_attn_bias);
    
    for (int l = 0; l < model->n_layers; l++) {
        struct ggml_tensor * residual = inpL;

        cur = build_t5_norm(ctx, inpL, model->layers[l].attn_norm);

        struct ggml_tensor * attn_out;

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx, model->layers[l].q, cur);
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx, model->layers[l].k, cur);
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx, model->layers[l].v, cur);

			Qcur = ggml_reshape_3d(ctx, Qcur, model->head_size, model->n_attn_heads, batch.n_tokens);
            Kcur = ggml_reshape_3d(ctx, Kcur, model->head_size, model->n_attn_heads, batch.n_tokens);

            struct ggml_tensor * q = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
            struct ggml_tensor * k = ggml_cont(ctx, ggml_permute(ctx, Kcur, 0, 2, 1, 3));

            struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
            kq = ggml_add(ctx, kq, pos_bias);

            kq = ggml_soft_max_ext(ctx, kq, KQ_mask_dec, 1.0f, 0.0f);

            struct ggml_tensor * v = ggml_cont_3d(ctx, ggml_transpose(ctx, Vcur), batch.n_tokens, model->head_size, model->n_attn_heads);
            struct ggml_tensor * kqv = ggml_mul_mat(ctx, kq, v);
            struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 2, 0, 1, 3);
            attn_out = ggml_cont_2d(ctx, kqv_merged, model->hidden_size, batch.n_tokens);
            attn_out = ggml_mul_mat(ctx, model->layers[l].o, attn_out);
        }

        cur = ggml_add(ctx, attn_out, residual);
        struct ggml_tensor * residualmlp = cur;

        // mlp
        {
        	cur = build_t5_norm(ctx, cur, model->layers[l].mlp_norm);
        	struct ggml_tensor * gate_proj = ggml_mul_mat(ctx, model->layers[l].wi_1, cur);
        	cur = ggml_mul(ctx, ggml_gelu(ctx, ggml_mul_mat(ctx, model->layers[l].wi_0, cur)), gate_proj);
        	cur = ggml_mul_mat(ctx, model->layers[l].wo, cur);
        }

		cur = ggml_add(ctx, cur, residualmlp);
        inpL = cur;
    }

    cur = build_t5_norm(ctx, cur, model->out_norm);

    if (model->down_proj) {
        cur = ggml_mul_mat(ctx, model->down_proj, cur);
    }

    if (model->down_proj_bias) {
        cur = ggml_add(ctx, cur, model->down_proj_bias);
    }

    ggml_build_forward_expand(gf, cur);

    free_build();
    
    return gf;
}

void t5_runner::set_inputs(t5_ubatch & batch) {
    ggml_backend_tensor_set(t5ctx->inp_tokens, batch.input_tokens, 0, batch.n_tokens*ggml_element_size(t5ctx->inp_tokens));
    float * attn_mask = nullptr;
    uint32_t * positions = nullptr;
    uint32_t * pos_bucket = nullptr;
    attn_mask = (float *) t5ctx->attn_mask->data;
    positions = (uint32_t *) t5ctx->positions->data;
    pos_bucket = (uint32_t *) t5ctx->inp_pos_bucket->data;
    int n_buckets = (int) model->relative_attn_buckets / 2;
    int max_exact = (int) n_buckets / 2;
	float logarithmic_denominator = log(128.0 / max_exact);
    for (int i = 0; i < batch.n_tokens; i++) {
        for (int ii = 0; ii < batch.n_tokens; ii++) {
        	int ab_rpos = abs(i - ii);
        	int rpos = i - ii;
            attn_mask[i*batch.n_tokens + ii] = 0.0f; //ii > i ? -INFINITY : 0.0f; 
            pos_bucket[i*batch.n_tokens + ii] = (uint32_t) (rpos > 0 ? n_buckets : 0) + (ab_rpos < max_exact ? ab_rpos : std::min((n_buckets - 1), (max_exact + (int)((log((ab_rpos / max_exact)) / logarithmic_denominator) * max_exact))));
        }
    }

}

void t5_runner::run(uint32_t * input_tokens, uint32_t sequence_length, struct t5_response * outputs) {
	t5_ubatch batch;
    batch.input_tokens = input_tokens;
    batch.n_tokens = sequence_length;
    ggml_backend_sched_reset(t5ctx->sched);
    
    const size_t prev_size = t5ctx->buf_output ? ggml_backend_buffer_get_size(t5ctx->buf_output) : 0;
    const size_t new_size = model->max_context_length * model->output_size * sizeof(float);
    
    if (!t5ctx->buf_output || prev_size < new_size) {
        if (t5ctx->buf_output) {
            ggml_backend_buffer_free(t5ctx->buf_output);
            t5ctx->buf_output = nullptr;
            t5ctx->logits = nullptr;
        }

        t5ctx->buf_output = ggml_backend_buft_alloc_buffer(t5ctx->backend_cpu_buffer, new_size);
    }
    
    outputs->encoded_states = (float *) ggml_backend_buffer_get_base(t5ctx->buf_output);
    ggml_backend_buffer_clear(t5ctx->buf_output, 0);
    struct ggml_cgraph * gf = NULL;
    gf = build_t5_graph(batch);
    // the output is always the last tensor in the graph
    struct ggml_tensor * result = gf->nodes[gf->n_nodes - 1];
    ggml_backend_sched_alloc_graph(t5ctx->sched, gf);
    set_inputs(batch);

    ggml_backend_sched_graph_compute_async(t5ctx->sched, gf);

    ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(t5ctx->sched, result);

    ggml_backend_tensor_get_async(backend_res, result, outputs->encoded_states, 0, batch.n_tokens*sizeof(float)*model->output_size);

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(t5ctx->sched);
    outputs->sequence_length = sequence_length;
    outputs->hidden_size = model->output_size;
    return;
}

struct t5_response * t5_runner::generate(const std::string prompt) {
	std::vector<uint32_t> tokens;
	tokenizer->tokenize(prompt, tokens);
	t5_response * resp = new t5_response;
    tokens.push_back(model->eos_token_id);

	run(tokens.data(), (uint32_t) tokens.size(), resp);
	return resp;
}
