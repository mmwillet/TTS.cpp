#include "parler.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <mutex>

void parler_context::reset(int32_t n_output_heads) {
    n_outputs = 0;
    prompt_end_position = 0;
    current_position = 0;
    output_size = 0;
    output_tokens.clear();
    eos_seen.clear();
    for (int i = 0; i < (int) n_output_heads; i++) {
        eos_seen.push_back(false);
    }
}

void parler_context::set_threads() {
    if (backend != nullptr) {
#ifdef GGML_METAL
        // this is form copied from llama.cpp, but has since been removed. I don't know if this should be tuned.
        ggml_backend_metal_set_n_cb(backend, 1);
#endif
    }
    if (backend_cpu != nullptr) {
        ggml_backend_cpu_set_n_threads(backend_cpu, n_threads);
        ggml_backend_cpu_set_threadpool(backend_cpu, threadpool);
    }
}

void parler_context::build_schedule() {
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

bool parler_context::prep_schedule(ggml_cgraph * gf) {
    return ggml_backend_sched_reserve(sched, gf);
}

struct parler_context * build_new_parler_context(struct parler_tts_model * model, int n_threads, bool use_cpu) {
    parler_context * pctx = new parler_context(model, n_threads);
    if (!use_cpu) {
#ifdef GGML_METAL
        pctx->backend = ggml_backend_metal_init();
#endif
    }
    pctx->eos_seen.reserve(model->n_output_heads);
    pctx->backend_cpu = ggml_backend_cpu_init();
    pctx->set_threads();
    pctx->build_schedule();
    pctx->buf_compute_meta.resize(ggml_tensor_overhead()*model->max_nodes() + ggml_graph_overhead_custom(model->max_nodes(), false));
    return pctx;
}

static bool parler_kv_cache_init(struct parler_kv_cache * cache, parler_tts_model * model, parler_context * pctx, int32_t seq_id) {
    const int64_t n_layer = (int64_t) model->layers.size();
    cache->seq_id = seq_id;
    
    ggml_backend_buffer_type_t buft = nullptr;
    // this will only really support cpu or metal for the time being;
    if (pctx->backend != nullptr) {
#ifdef GGML_METAL
        buft = ggml_backend_metal_buffer_type();
#endif
    } else {
        buft = ggml_backend_cpu_buffer_type();
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 2u*model->n_layers*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    cache->ctx = ctx;
    

    cache->k_l.reserve(n_layer);
    cache->v_l.reserve(n_layer);

    for (int i = 0; i < (int) n_layer; i++) {
        ggml_tensor * k = ggml_new_tensor_1d(cache->ctx, cache->type_k, model->hidden_size*model->max_ctx_length);
        ggml_tensor * v = ggml_new_tensor_1d(cache->ctx, cache->type_v, model->hidden_size*model->max_ctx_length);
        ggml_format_name(k, "cache_k_l%d", i);
        ggml_format_name(v, "cache_v_l%d", i);
        cache->k_l.push_back(k);
        cache->v_l.push_back(v);
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(cache->ctx, buft);
    if (!buf) {
        return false;
    }
    ggml_backend_buffer_clear(buf, 0);
    cache->buf = buf;

    return true;
}

struct ggml_tensor * parler_build_inp_embd(struct ggml_context * ctx, struct parler_context * pctx, parler_tts_model * model, parler_ubatch & batch) {
    // Parler has two embedding schemas one for the text input and one for generative audio tokens. These two schemas have effectively distinct shapes (i.e. [batch_size, sequence_length] and [batch_size, sequence_lenghth, num_codebooks] respectively).
    // This means that depending on where we are in generation we need to follow a distinct pattern
    struct ggml_tensor * input_embs;
    pctx->positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.sequence_length);
    ggml_set_input(pctx->positions);
    if (batch.audio_generation) {
        pctx->audio_inp_tokens = ggml_reshape_2d(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_audio_tokens), batch.n_audio_tokens / model->n_output_heads, model->n_output_heads);
        ggml_set_input(pctx->audio_inp_tokens);
        struct ggml_tensor * audio_tokens = ggml_reshape_2d(ctx, pctx->audio_inp_tokens, batch.n_audio_tokens / model->n_output_heads, model->n_output_heads);
        for (int i = 0; i < model->n_output_heads; i++) {
            if (i == 0) {
                input_embs = ggml_get_rows(ctx, model->embds[i], ggml_view_2d(ctx, audio_tokens, 1, batch.n_audio_tokens / model->n_output_heads, audio_tokens->nb[1], i*sizeof(int32_t)));
            } else {
                input_embs = ggml_add(ctx, ggml_get_rows(ctx, model->embds[i], ggml_view_2d(ctx, audio_tokens, 1, batch.n_audio_tokens / model->n_output_heads, audio_tokens->nb[1], i*sizeof(int32_t))), input_embs);
            }
        }
    } else {
        pctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
        ggml_set_input(pctx->inp_tokens);
        input_embs = ggml_get_rows(ctx, model->prompt_embd, pctx->inp_tokens);
    }
    return ggml_add(ctx, input_embs, ggml_get_rows(ctx, model->precomputed_positional_embds, pctx->positions));
}

struct ggml_tensor * parler_build_layer_norm(struct ggml_context * ctx, struct ggml_tensor * inputs, struct ggml_tensor * weight, struct ggml_tensor * bias) {
    // parler always uses default eps
    float eps = 0.00001;
    inputs = ggml_norm(ctx, inputs, eps);
    inputs = ggml_mul(ctx, inputs, weight);
    return ggml_add(ctx, inputs, bias);
}

void parler_build_kv_store(struct ggml_context * ctx, parler_kv_cache * kv, struct ggml_cgraph * graph, struct ggml_tensor * k_cur, struct ggml_tensor * v_cur, int32_t n_tokens, int32_t kv_head, int32_t index, int32_t n_embd_gqa) {
    const int64_t n_ctx = 4096;

    struct ggml_tensor * k_cache_view = ggml_view_1d(ctx, kv->k_l[index], n_tokens*n_embd_gqa, ggml_row_size(kv->k_l[index]->type, n_embd_gqa)*kv_head);

    ggml_build_forward_expand(graph, ggml_cpy(ctx, k_cur, k_cache_view));

    assert(v_cur->ne[0] == n_embd_gqa && v_cur->ne[1] == n_tokens);

    struct ggml_tensor * v_cache_view = nullptr;

    v_cache_view = ggml_view_2d(ctx, kv->v_l[index], n_tokens, n_embd_gqa,
            (  n_ctx)*ggml_element_size(kv->v_l[index]),
            (kv_head)*ggml_element_size(kv->v_l[index]));

    v_cur = ggml_transpose(ctx, v_cur);

    ggml_build_forward_expand(graph, ggml_cpy(ctx, v_cur, v_cache_view));
}

struct ggml_tensor * parler_build_head_outputs(struct ggml_context * ctx, parler_tts_model * model, struct ggml_tensor * cur) {
    // going to cat the heads together and then reshape them;
    // honestly ggml doesn't provide good support for stacking and discrete tensor access
    struct ggml_tensor * out;
    for (int i = 0; i < model->n_output_heads; i++) {
        if (i == 0) {
            out = ggml_mul_mat(ctx, model->heads[i], cur);
        } else {
            out = ggml_concat(ctx, out, ggml_mul_mat(ctx, model->heads[i], cur), 1);
        }
    }
    ggml_set_name(out, "final_out");
    //out = ggml_cont(ctx, ggml_transpose(ctx, out));

    int32_t sql_len = (int32_t) (ggml_nelements(out) / (model->output_vocab_size * model->n_output_heads));
    return ggml_cont_3d(ctx, out, model->output_vocab_size, sql_len, model->n_output_heads);
}

struct ggml_tensor * build_attn_mask(ggml_context * ctx, parler_context * pctx, parler_ubatch & batch) {
    pctx->attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) pctx->current_position + batch.sequence_length, (int64_t) pctx->current_position + batch.sequence_length);
    ggml_set_input(pctx->attn_mask);

    return pctx->attn_mask;
}

struct ggml_tensor * build_attn_mask_cross(ggml_context * ctx, parler_context * pctx, parler_tts_model * model, parler_ubatch & batch) {
    pctx->attn_mask_cross = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) model->n_encode_length, (int64_t) batch.sequence_length);
    ggml_set_input(pctx->attn_mask_cross);
    
    return pctx->attn_mask_cross;
}

static struct parler_ubatch batch_from_sentence(std::string sentence, parler_tts_model * model, unigram_tokenizer * tokenizer) {
    struct parler_ubatch batch;
    batch.audio_generation = false;
    std::vector<uint32_t>* token_ids = new std::vector<uint32_t>;
    tokenizer->tokenize(sentence, *token_ids);
    token_ids->push_back(tokenizer->eos_token);
    batch.current_step = 0;
    batch.n_tokens = token_ids->size();
    batch.n_audio_tokens = 0;
    batch.sequence_length = batch.n_tokens; // sequence_length is equal to the number of tokens for non-audio generation
    std::vector<uint32_t>* position = new std::vector<uint32_t>;
    for (uint32_t i = 0; i < batch.sequence_length; i++) {
        position->push_back(i);
    }
    std::vector<uint32_t>* order = new std::vector<uint32_t>;
    for (int i = 0; i < batch.sequence_length; i++) {
        if (i >= batch.sequence_length - 1) {
            order->push_back(0);
        } else {
            order->push_back(i+1);
        }
    }
    batch.positions = position->data();
    batch.tokens = token_ids->data();
    return batch;
}
        
void parler_tts_runner::init_build() {
    struct ggml_init_params params = {
        /*.mem_size   =*/ pctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ pctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx = ggml_init(params);

    pctx->inp_tokens       = nullptr;
    pctx->audio_inp_tokens = nullptr;
}

void parler_tts_runner::free_build() {
    if (ctx) {
        ggml_free(ctx);
        ctx = nullptr;
    }
}

struct ggml_cgraph * parler_tts_runner::build_parler_graph(parler_ubatch & batch) {
    init_build();
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    
    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;
    
    const int32_t full_sequence_length = pctx->current_position + (uint32_t) batch.sequence_length;
    
    inpL = parler_build_inp_embd(ctx, pctx, model, batch);
    
    struct ggml_tensor * KQ_mask_dec = build_attn_mask(ctx, pctx, batch);
    struct ggml_tensor * KQ_mask_cross = build_attn_mask_cross(ctx, pctx, model, batch);
    
    for (int l = 0; l < model->n_layers; l++) {
        struct ggml_tensor * residual = inpL;
        ggml_set_name(inpL, ("layer_" + std::to_string(l) + "_input").c_str());
        // each layer goes ->
        // self-attn-norm -> self-attn -> res + out -> encoder-attn-norm -> res + out -> final-norm -> fc1 -> act -> fc2 -> res + out ->

        cur = parler_build_layer_norm(ctx, inpL, model->layers[l]->self_attn_norm, model->layers[l]->self_attn_norm_bias);

        struct ggml_tensor * attn_out;

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx, model->layers[l]->self_attn_q_proj, cur);
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx, model->layers[l]->self_attn_k_proj, cur);
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx, model->layers[l]->self_attn_v_proj, cur);

            parler_build_kv_store(ctx, kv_self, gf, Kcur, Vcur, (int32_t) batch.sequence_length, pctx->current_position, l, model->hidden_size);
            struct ggml_tensor * k =
                ggml_view_3d(ctx, kv_self->k_l[l],
                        model->head_size, full_sequence_length, model->n_attn_heads,
                        ggml_row_size(kv_self->k_l[l]->type, model->hidden_size),
                        ggml_row_size(kv_self->k_l[l]->type, model->head_size),
                        0);
            
            
            struct ggml_tensor * v =
                ggml_view_3d(ctx, kv_self->v_l[l],
                        full_sequence_length, model->head_size, model->n_attn_heads,
                        ggml_element_size(kv_self->v_l[l])*model->max_ctx_length,
                        ggml_element_size(kv_self->v_l[l])*model->max_ctx_length*model->head_size,
                        0);
                        
            Qcur = ggml_reshape_3d(ctx, Qcur, model->head_size, model->n_attn_heads, batch.sequence_length);
            struct ggml_tensor * q = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
            struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
            kq = ggml_soft_max_ext(ctx, kq, KQ_mask_dec, 0.125f, 0.0f);
            struct ggml_tensor * kqv = ggml_mul_mat(ctx, kq, v);
            struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 2, 0, 1, 3);
            attn_out = ggml_cont_2d(ctx, kqv_merged, model->hidden_size, batch.sequence_length);
            attn_out = ggml_mul_mat(ctx, model->layers[l]->self_attn_o_proj, attn_out);
        }

        cur = ggml_add(ctx, attn_out, residual);
        struct ggml_tensor * residualffn = cur;
        
        if (model->use_cross_attn) {
            struct ggml_tensor * residuala = cur;

            // norm
            cur = parler_build_layer_norm(ctx, cur, model->layers[l]->attn_norm, model->layers[l]->attn_norm_bias);
            struct ggml_tensor * cross_attn_out;

            //cross-attention
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx, model->layers[l]->attn_q_proj, cur);
            Qcur = ggml_reshape_3d(ctx, Qcur,  model->head_size, model->n_attn_heads, batch.sequence_length);
            
            struct ggml_tensor * q = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
            
            struct ggml_tensor * kq = ggml_mul_mat(ctx, model->layers[l]->cross_k, q);
            kq = ggml_soft_max_ext(ctx, kq, KQ_mask_cross, 0.125f, 0.0f);
             
            struct ggml_tensor * kqv  = ggml_mul_mat(ctx, kq, model->layers[l]->cross_v);
            struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 2, 0, 1, 3);
            cross_attn_out = ggml_cont_2d(ctx, kqv_merged, model->hidden_size, batch.sequence_length);
            cross_attn_out = ggml_mul_mat(ctx, model->layers[l]->attn_o_proj, cross_attn_out);

            residualffn = ggml_add(ctx, cross_attn_out, residuala);
        }

        cur = parler_build_layer_norm(ctx, residualffn, model->layers[l]->final_norm, model->layers[l]->final_norm_bias);
        cur = ggml_mul_mat(ctx, model->layers[l]->fc1, cur);
        cur = ggml_gelu(ctx, cur);
        cur = ggml_mul_mat(ctx, model->layers[l]->fc2, cur);
        cur = ggml_add(ctx, cur, residualffn);
        inpL = cur;
    }
    
    cur = parler_build_layer_norm(ctx, cur, model->layer_norm, model->layer_norm_bias);
    cur = parler_build_head_outputs(ctx, model, cur);
    ggml_build_forward_expand(gf, cur);
    free_build();
    
    return gf;
}

void parler_tts_runner::set_inputs(parler_ubatch & batch) {
    if (batch.audio_generation) {
        ggml_backend_tensor_set(pctx->audio_inp_tokens, batch.audio_tokens, 0, batch.n_audio_tokens*ggml_element_size(pctx->audio_inp_tokens));
    } else {
        ggml_backend_tensor_set(pctx->inp_tokens, batch.tokens, 0, batch.n_tokens*ggml_element_size(pctx->inp_tokens));
    }
    ggml_backend_tensor_set(pctx->positions, batch.positions, 0, batch.sequence_length*ggml_element_size(pctx->positions));
    float * d = nullptr;
    d = (float *) pctx->attn_mask->data;
    uint32_t max_pos = pctx->current_position + batch.sequence_length;
    for (int i = 0; i < batch.sequence_length; i++) {
        uint32_t pos = batch.positions[i];
        for (int ii = 0; ii < max_pos; ii++) {
            d[i*max_pos + ii] = ii > pos ? -INFINITY : 0.0f;
        }
    }
    
    if (model->use_cross_attn) {
        float * d2 = nullptr;
        d2 = (float *) pctx->attn_mask_cross->data;
        for (int i = 0; i < model->n_encode_length; i++) {
            for (int ii = 0; ii < batch.sequence_length; ii++) {
                d2[i*batch.sequence_length + ii] = 0.0f;
            }
        }
    }

}
void parler_tts_runner::parler_graph_compute(ggml_cgraph * gf) {
    ggml_backend_sched_graph_compute_async(pctx->sched, gf);
}

int parler_tts_runner::decode(parler_ubatch & batch) {
    ggml_backend_sched_reset(pctx->sched);
    
    pctx->output_tokens.reserve(model->max_generation_size);
    
    const size_t logits_size = model->output_vocab_size*model->max_generation_size*model->n_output_heads;
    const size_t prev_size = pctx->buf_output ? ggml_backend_buffer_get_size(pctx->buf_output) : 0;
    const size_t new_size  = logits_size * sizeof(float);
    
    if (!pctx->buf_output || prev_size < new_size) {
        if (pctx->buf_output) {
            ggml_backend_buffer_free(pctx->buf_output);
            pctx->buf_output = nullptr;
            pctx->logits = nullptr;
        }

        pctx->buf_output = ggml_backend_buft_alloc_buffer(pctx->backend_cpu_buffer, new_size);
    }
    
    pctx->logits = (float *) ggml_backend_buffer_get_base(pctx->buf_output);
    //ggml_backend_buffer_clear(pctx->buf_output, 0);

    ggml_cgraph * gf = build_parler_graph(batch);

    // the output is always the last tensor in the graph
    struct ggml_tensor * res = gf->nodes[gf->n_nodes - 1];
    ggml_backend_sched_alloc_graph(pctx->sched, gf);
    
    // use the sequence_length variable here so that audio input tokens are handled correctly.
    size_t n_outputs_new = batch.sequence_length;

    set_inputs(batch);
    parler_graph_compute(gf);
    ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(pctx->sched, res);
    
    float * logits_out = pctx->logits + pctx->n_outputs * model->output_vocab_size * model->n_output_heads;
    ggml_backend_tensor_get_async(backend_res, res, logits_out, 0, n_outputs_new*model->output_vocab_size*model->n_output_heads*sizeof(float));

    // set to total number of outputs in the batch*/
    pctx->n_outputs += n_outputs_new;

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(pctx->sched);

    return 0;
}

parler_ubatch parler_tts_runner::build_worst_case_batch()  {
    struct parler_ubatch batch;
    batch.audio_generation = false;
    batch.n_tokens = model->max_ctx_length;
    batch.n_audio_tokens = 0;
    batch.sequence_length = model->max_ctx_length;
    return batch;
}

void parler_tts_runner::prepare_post_load() {
    parler_kv_cache_init(kv_self, model, pctx, std::mt19937(std::random_device{}())());
    auto batch = build_worst_case_batch();
    auto gf = build_parler_graph(batch);
    pctx->prep_schedule(gf);
}

bool parler_tts_runner::adjust_for_sequence_continuation(struct parler_ubatch & batch) {
    return false; // not implemneted
}

bool parler_tts_runner::check_stopping() {
    int32_t token_position = (int32_t) pctx->output_tokens.size() - (int32_t) model->n_output_heads;
    if (token_position < 0) {
        return false;
    }
    if (pctx->current_position >= model->max_generation_size) {
        return true;
    }
        
    bool channels_complete = true;
    for (int i = 0; i < model->n_output_heads; i++) {
        pctx->eos_seen[i] = pctx->eos_seen[i] || pctx->output_tokens[token_position+i] == model->eos_token_id;
        if (channels_complete) {
            channels_complete = pctx->eos_seen[i];
        }
    }
    return channels_complete;
}

void parler_tts_runner::adjust_output_tokens(std::vector<uint32_t> & output_tokens, std::vector<uint32_t> & filtered) {
    // currently this is applying sliding window over the heads and filtering out bad tokens.
    // If we convert the DAC model's quantizer layers to support by row + column embeddings then we will need to transpose
    // the heads and the sequence here, but right now simplying using a strided view is more peformant.
    size_t size = output_tokens.size();
    filtered.reserve(size);
    for (int i = 0; i < size / model->n_output_heads; i++) {
        bool remove = false;
        for (int ii = 0; ii < model->n_output_heads; ii++) {
            int next_index = i*model->n_output_heads+ii*model->n_output_heads+ii;
            if (next_index > size || output_tokens[next_index] >= model->audio_vocab_size) {
                remove = true;
                break;
            }
        }
        if (!remove) {
            for (int ii = 0; ii < model->n_output_heads; ii++) {
                int next_index = i*model->n_output_heads+ii*model->n_output_heads+ii;
                if (next_index > size) {
                    filtered.push_back(model->eos_token_id);
                } else {
                    filtered.push_back(output_tokens[next_index]);
                }
            }
        }
    }
}

int parler_tts_runner::generate_from_batch(parler_ubatch & batch, struct tts_response * output) {
    std::vector<uint32_t> next_decoder_token_ids;
    next_decoder_token_ids.reserve(model->n_output_heads);

    while (!check_stopping()) {
        int state = decode(batch);
        if (state != 0) {
            return state;
        }
        if (!batch.audio_generation) {
            pctx->prompt_end_position += batch.sequence_length;
        }
        if (batch.audio_generation) {
            sampler->sample(pctx->logits + pctx->current_position * model->n_output_heads * model->output_vocab_size, pctx->output_tokens);
        }
        pctx->current_position += batch.sequence_length;
        next_decoder_token_ids.clear();
        uint32_t * last_outputs = (pctx->output_tokens.data() + (int) pctx->output_tokens.size() - model->n_output_heads);
        for (int i = 0; i < model->n_output_heads; i++) {
            next_decoder_token_ids.push_back(batch.current_step > i ? pctx->eos_seen[i] ? model->eos_token_id : last_outputs[i] : model->bos_token_id);
        }
        batch = parler_ubatch{
            true, 0, 9, 1, nullptr, next_decoder_token_ids.data(), &pctx->current_position, nullptr, batch.current_step+1
        };
    }


    std::vector<uint32_t> filtered_output_tokens;
    adjust_output_tokens(pctx->output_tokens, filtered_output_tokens);
    dac_runner->run(filtered_output_tokens.data(), (int32_t) filtered_output_tokens.size() / model->n_output_heads, output);
    return 0;
}

int parler_tts_runner::generate_audio_tokens(std::string sentence) {
    parler_ubatch batch = batch_from_sentence(sentence, model, tokenizer);
    pctx->reset(model->n_output_heads);
    sampler->reset();
    int32_t seq_id = std::mt19937(std::random_device{}())();
    if (!kv_self) {
        kv_self = new parler_kv_cache;
        if (!parler_kv_cache_init(kv_self, model, pctx, seq_id)) {
            return 1;
        }
    }

    std::vector<uint32_t> next_decoder_token_ids;
    next_decoder_token_ids.reserve(model->n_output_heads);

    while (!check_stopping()) {
        int state = decode(batch);
        if (state != 0) {
            return state;
        }
        if (!batch.audio_generation) {
            pctx->prompt_end_position += batch.sequence_length;
        }
        if (batch.audio_generation) {
            sampler->sample(pctx->logits + pctx->current_position * model->n_output_heads * model->output_vocab_size, pctx->output_tokens);
        }
        pctx->current_position += batch.sequence_length;
        next_decoder_token_ids.clear();
        uint32_t * last_outputs = (pctx->output_tokens.data() + (int) pctx->output_tokens.size() - model->n_output_heads);
        for (int i = 0; i < model->n_output_heads; i++) {
            next_decoder_token_ids.push_back(batch.current_step > i ? pctx->eos_seen[i] ? model->eos_token_id : last_outputs[i] : model->bos_token_id);
        }
        batch = parler_ubatch{
            true, 0, 9, 1, nullptr, next_decoder_token_ids.data(), &pctx->current_position, nullptr, batch.current_step+1
        };
    }

    return 0;
}

void parler_tts_runner::just_audio_token_decode(uint32_t * tokens, int32_t sq_len, struct tts_response * outputs) {
    dac_runner->run(tokens, sq_len, outputs);
}

int parler_tts_runner::generate(std::string sentence, struct tts_response * output, int32_t seq_id) {
    parler_ubatch batch = batch_from_sentence(sentence, model, tokenizer);
    pctx->reset(model->n_output_heads);
    sampler->reset();
    if (pctx->seq_id != seq_id || seq_id == -1) {
        seq_id = std::mt19937(std::random_device{}())();
        pctx->current_position = 0;
        if (!kv_self) {
            kv_self = new parler_kv_cache;
            if (!parler_kv_cache_init(kv_self, model, pctx, seq_id)) {
                return 1;
            }
        }
    } else {
        if (!adjust_for_sequence_continuation(batch)) {
            return 2;
        }
    }
    return generate_from_batch(batch, output);
}

// currently only metal and cpu devices are supported, so cpu_only only describes whether or not to try to load and run on metal.
struct parler_tts_runner * runner_from_file(const std::string & fname, int n_threads, bool cpu_only, bool use_cross_attn) {
    parler_tts_model * model = new parler_tts_model;
    dac_model * audio_model = new dac_model;
    ggml_context * weight_ctx = NULL;

    struct gguf_init_params params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &weight_ctx,
    };
    gguf_context * meta_ctx = gguf_init_from_file(fname.c_str(), params);
    if (!meta_ctx) {
        TTS_ABORT("%s failed for file %s\n", __func__, fname.c_str());
    }
    unigram_tokenizer * ut = tokenizer_from_gguf(meta_ctx);
    ut->initialize_tokenizer();
    model->use_cross_attn = use_cross_attn;
    model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
    audio_model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
    
    // TODO: change this weight assignment pattern to mirror llama.cpp
    for (ggml_tensor * cur = ggml_get_first_tensor(weight_ctx); cur; cur = ggml_get_next_tensor(weight_ctx, cur)) {
        assign_weight(model, *audio_model, cur->name, cur);
    }
    if (use_cross_attn) {
        model->prep_cross_key_values();
    }
    
    struct dac_context * dctx = build_new_dac_context(audio_model, n_threads, cpu_only);
    struct dac_runner * audio_decoder = new dac_runner(audio_model, dctx);
    audio_decoder->prepare_post_load();
    struct sampler * samp = new sampler;
    struct parler_context * pctx = build_new_parler_context(model, n_threads, cpu_only);
    struct parler_kv_cache * cache = new parler_kv_cache;
    
    struct parler_tts_runner * runner = new parler_tts_runner(model, audio_decoder, pctx, ut, samp, cache);//, weight_ctx);
    runner->prepare_post_load();
    gguf_free(meta_ctx);
    ggml_free(weight_ctx);
    return runner;
}

bool is_quanitizable(std::string name, struct quantization_params * params) {
    // the DAC audio encoder / decoder is not compatible with quantization, normalization weight shouldn't be quantized, and the text encoding shouldn't be normalized.
    bool quantizable = !has_prefix(name, "audio_encoder") && !has_suffix(name, "norm.weight") && !has_suffix(name, "text_encoding") && !has_suffix(name, "positional_embed") && !has_suffix(name, "norm.bias");
    if (!params->quantize_output_heads) {
        quantizable = quantizable && !has_suffix(name, "weight.head");
    }
    if (!params->quantize_text_embeddings) {
        quantizable = quantizable && !has_suffix(name, "embed_prompts");
    }
    if (!params->quantize_cross_attn_kv) {
        quantizable = quantizable && !has_suffix(name, "encoder_attn.k_proj.weight") && !has_suffix(name, "encoder_attn.v_proj.weight");   
    }
    return quantizable;
}

size_t quantize_tensor(void * new_data, struct ggml_tensor * tensor, const float * imatrix, enum ggml_type qtype, uint32_t n_threads) {
    // much of this is form copied from llama.cpp
    int chunk_size_multiplier = 1;
    if (qtype == GGML_TYPE_Q4_0_4_4 || qtype == GGML_TYPE_Q4_0_4_8 || qtype == GGML_TYPE_Q4_0_8_8) {
        if ((qtype == GGML_TYPE_Q4_0_8_8) && (tensor->ne[1] % 8 != 0)) qtype = GGML_TYPE_Q4_0;
        else if (tensor->ne[1] % 4 != 0) qtype = GGML_TYPE_Q4_0;
        if (qtype == GGML_TYPE_Q4_0_8_8) chunk_size_multiplier = 8;
        else if (qtype == GGML_TYPE_Q4_0_4_4 || qtype == GGML_TYPE_Q4_0_4_8) chunk_size_multiplier = 4;
    }
    size_t out_size = 0;
    const int32_t d3_step = tensor->ne[0] * tensor->ne[1];
    const int32_t n_per_row = tensor->ne[0];
    const int32_t nrows = tensor->ne[1];
    static const int32_t min_chunk_size = 32 * 512;
    const int32_t chunk_size = (n_per_row >= min_chunk_size ? n_per_row : n_per_row * ((min_chunk_size + n_per_row - 1)/n_per_row)) * chunk_size_multiplier;
    uint32_t thread_count = std::max(1, std::min((int)n_threads, (int)(d3_step + chunk_size - 1) / chunk_size));
    std::mutex mutex;

    for (int32_t d3_index = 0; d3_index < tensor->ne[2]; d3_index++) {
        const float * f32_data_d3 = ((float *) tensor->data) + d3_index * d3_step;
        void * new_data_d3 = (char *)new_data + ggml_row_size(qtype, tensor->ne[0]) * d3_index * nrows;
        const float * imatrix_03 = imatrix ? imatrix + d3_index * tensor->ne[0] : nullptr;
        if (thread_count <= 1) {
            // not threaded
            out_size += ggml_quantize_chunk(qtype, f32_data_d3, new_data_d3, 0, nrows, n_per_row, imatrix);
        } else {
            std::vector <std::thread> threads;
            int64_t counter = 0;
            size_t new_size = 0;
            bool valid = true;
            for (uint32_t t = 0; t < thread_count; t++) {
                auto func = [&mutex, &counter, &new_size, &valid, qtype, f32_data_d3, new_data_d3, chunk_size, nrows, n_per_row, imatrix]() {
                    const int64_t nrows_per_chunk = chunk_size / n_per_row;
                    size_t local_size = 0;
                    while (true) {
                        std::unique_lock<std::mutex> lock(mutex);
                        int64_t first_row = counter; counter += nrows_per_chunk;
                        if (first_row >= nrows) {
                            if (local_size > 0) {
                                new_size += local_size;
                            }
                            break;
                        }
                        lock.unlock();
                        const int64_t this_nrow = std::min(nrows - first_row, nrows_per_chunk);
                        size_t this_size = ggml_quantize_chunk(qtype, f32_data_d3, new_data_d3, first_row * n_per_row, this_nrow, n_per_row, imatrix);
                        local_size += this_size;

                        // validate the quantized data; I am not sure how this would occur, but there is always the safe fallback on doing this single threaded.
                        const size_t row_size  = ggml_row_size(qtype, n_per_row);
                        void * this_data = (char *) new_data_d3 + first_row * row_size;
                        if (!ggml_validate_row_data(qtype, this_data, this_size)) {
                            std::unique_lock<std::mutex> lock(mutex);
                            valid = false;
                            break;
                        }
                    }
                };
                threads.push_back(std::thread(func));
            }
            for (auto & t : threads) t.join();

            if (!valid) {
                TTS_ABORT("Validation of quantized data failed. Please try again and/or switch to single thread quantization.\n");
            }
            out_size += new_size;
        }
    }
    return out_size;
}

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

template <typename T>
struct no_init {
    T value;
    no_init() { /* do nothing */ }
};

void quantize_gguf(const std::string & ifile, const std::string & ofile, struct quantization_params * params) {
    ggml_context * weight_ctx = NULL;
    struct gguf_init_params gguf_params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &weight_ctx,
    };
    gguf_context * meta_ctx = gguf_init_from_file(ifile.c_str(), gguf_params);

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    gguf_context_ptr ctx_out { gguf_init_empty() };

    // copy the KV pairs from the input file
    gguf_set_kv(ctx_out.get(), meta_ctx);
    gguf_set_val_u32(ctx_out.get(), "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out.get(), "general.quantization_type", params->quantize_type);

    for (ggml_tensor * tensor = ggml_get_first_tensor(weight_ctx); tensor; tensor = ggml_get_next_tensor(weight_ctx, tensor)) {
        gguf_add_tensor(ctx_out.get(), tensor);
    }

    std::vector<no_init<uint8_t>> work;

    const std::unordered_map<std::string, std::vector<float>> * imatrix_data = nullptr;
    if (params->imatrix) {
        imatrix_data = static_cast<const std::unordered_map<std::string, std::vector<float>>*>(params->imatrix);
        if (imatrix_data) {
            // check imatrix for nans or infs
            for (const auto & kv : *imatrix_data) {
                for (float f : kv.second) {
                    if (!std::isfinite(f)) {
                        TTS_ABORT("ERROR: imatrix contains non-finite value %f\n", f);
                    }
                }
            }
        }
    }

    std::ofstream fout;
    auto close_ofstream = [&]() {
        // Write metadata and close file handler
        if (fout.is_open()) {
            fout.seekp(0);
            std::vector<uint8_t> data(gguf_get_meta_size(ctx_out.get()));
            gguf_get_meta_data(ctx_out.get(), data.data());
            fout.write((const char *) data.data(), data.size());
            fout.close();
        }
    };
    auto new_ofstream = [&]() {
        std::string fname = ofile;
        fout = std::ofstream(fname, std::ios::binary);
        fout.exceptions(std::ofstream::failbit); // fail fast on write errors
        const size_t meta_size = gguf_get_meta_size(ctx_out.get());
        // placeholder for the meta data
        ::zeros(fout, meta_size);
    };
    new_ofstream();
    for (ggml_tensor * cur = ggml_get_first_tensor(weight_ctx); cur; cur = ggml_get_next_tensor(weight_ctx, cur)) {
        enum ggml_type new_type;
        void * new_data;
        size_t new_size;
        std::string name = ggml_get_name(cur);
        if (name.size() != 0 && is_quanitizable(name, params)) {
            if ((cur->type) != GGML_TYPE_F32) {
                TTS_ABORT("ERROR: All quantized tensors must be transformed from 32bit floats. Tensor, '%s', has impropert type, '%d'\n", cur->name, cur->type);
            }
            new_type = params->quantize_type;
            if ((new_type >= GGML_TYPE_IQ2_XXS && new_type <= GGML_TYPE_IQ4_XS) && imatrix_data) {
                TTS_ABORT("ERROR: Quantization type '%d' requires an importance matrix.\n", new_type);
            }
            const float * imatrix = nullptr;
            if (imatrix_data) {
                auto it = imatrix_data->find(cur->name);
                if (it == imatrix_data->end()) {
                    TTS_ABORT("ERROR: importance matrix does not contain information for tensor, '%s.'\n", cur->name);
                } else {
                    imatrix = it->second.data();
                }
            }
            const int64_t nelement_size = ggml_nelements(cur) * 4;
            if (work.size() < (size_t)nelement_size) {
                work.resize(nelement_size); // upper bound on size
            }
            new_data = work.data();
            new_size = quantize_tensor(new_data, cur, imatrix, new_type, params->n_threads);
        } else if (params->convert_dac_to_f16 && has_prefix(name, "audio_encoder") && !has_suffix(name, "alpha") && !has_suffix(name, "bias")) {
            if ((cur->type) != GGML_TYPE_F32) {
                TTS_ABORT("ERROR: All converted tensors must be transformed from 32bit floats. Tensor, '%s', has impropert type, '%d'\n", cur->name, cur->type);
            }
            new_type = GGML_TYPE_F16;
            const int64_t nelement_size = ggml_nelements(cur) * 4;
            if (work.size() < (size_t)nelement_size) {
                work.resize(nelement_size); // upper bound on size
            }
            new_data = work.data();
            new_size = quantize_tensor(new_data, cur, nullptr, new_type, params->n_threads);
        } else {
            new_type = cur->type;
            new_data = cur->data;
            new_size = ggml_nbytes(cur);
        }

        gguf_set_tensor_type(ctx_out.get(), name.c_str(), new_type);
        gguf_set_tensor_data(ctx_out.get(), name.c_str(), new_data, new_size);
        fprintf(stdout, "At tensor: '%s' with new size: %d bytes\n", name.c_str(), new_size);
        // write tensor data + padding
        fout.write((const char *) new_data, new_size);
        zeros(fout, GGML_PAD(new_size, align) - new_size);
    }
    close_ofstream();
}
