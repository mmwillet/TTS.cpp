#include <algorithm>
#include <map>
#include <ranges>
#include <stdexcept>
#include "dac_model.h"

#include "tts_ggml_iterator.h"

dac_model_constants::dac_model_constants(gguf_context * meta) : layer_constants(n_layers) {
    static constexpr array<pair<str, const uint32_t dac_model_constants::*>, 8> OFFSETS{{
        {"parler-tts.decoder.output_heads", &dac_model::n_heads},
        {"output_heads", &dac_model::n_heads},
        {"dia.decoder.output_heads", &dac_model::n_heads},
        {"dac.up_sampling_factor", &dac_model::up_sampling_factor},
        {"up_sampling_factor", &dac_model::up_sampling_factor},
        {"parler-tts.decoder.max_generation", &dac_model::max_generation_size},
        {"max_generation", &dac_model::max_generation_size},
        {"dia.decoder.max_generation", &dac_model::max_generation_size},
    }};
    static_assert(is_sorted(OFFSETS.begin(), OFFSETS.end()));
    for (const auto [i, key] : gguf_key_iterator{*meta}) {
        sv k{key};
        if (const auto found{binary_search_idx(OFFSETS, k, pair_first_cmp_sv)}; ~found) {
            const_cast<uint32_t &>(this->*OFFSETS[found].second) = gguf_get_val_u32(meta, i);
        } else if (k.starts_with("dac.dac_layer_")) {
            k.remove_prefix(sizeof("dac.dac_layer_") - 1);
            if (k.starts_with("stride_")) {
                k.remove_prefix(sizeof("stride_") - 1);
                const_cast<uint32_t &>(layer_constants.at(sv_int(k)).stride) = gguf_get_val_u32(meta, i);
            } else if (k.starts_with("padding_")) {
                k.remove_prefix(sizeof("padding_") - 1);
                const_cast<uint32_t &>(layer_constants.at(sv_int(k)).stride) = gguf_get_val_u32(meta, i);
            }
        }
    }
    for (const auto [padding, stride] : layer_constants) {
        TTS_ASSERT(~stride);
        TTS_ASSERT(~padding);
    }
}

void dac_residual_unit::assign_weight(ggml_context * ctx, sv name, ggml_tensor * tensor) {
    static constexpr array<tuple<str, ggml_tensor * dac_residual_unit::*, bool>, 6> OFFSETS{{
        {".res.final.alpha", &dac_residual_unit::out_snake_alpha, false},
        {".res.final.bias", &dac_residual_unit::out_conv_bias, true},
        {".res.final.weight", &dac_residual_unit::out_conv_kernel, false},
        {".res.initial.alpha", &dac_residual_unit::in_snake_alpha, false},
        {".res.initial.bias", &dac_residual_unit::in_conv_bias, true},
        {".res.initial.weight", &dac_residual_unit::in_conv_kernel, false},
    }};
    const auto found{binary_search_idx(OFFSETS, name, pair_first_cmp_sv)};
    TTS_ASSERT(~found);
    const auto [_, member, transpose] = OFFSETS[found];
    this->*member = ggml_dup_tensor(ctx, !transpose ? tensor : ggml_transpose(ctx, tensor));
}

void dac_layer::assign_weight(ggml_context * ctx, sv name, ggml_tensor * tensor) {
    static constexpr array<tuple<str, ggml_tensor * dac_layer::*, bool>, 3> OFFSETS{{
        {".final.alpha", &dac_layer::snake_alpha_in, false},
        {".final.bias", &dac_layer::out_conv_bias, true},
        {".final.weight", &dac_layer::out_conv_kernel, false},
    }};
    for (const auto [k, member, transpose] : OFFSETS) {
        if (k == name) {
            this->*member = ggml_dup_tensor(ctx, !transpose ? tensor : ggml_transpose(ctx, tensor));
            return;
        }
    }
    if (const auto [index, lt_name] = parse_layer_count(name); ~index) {
        residual_blocks[index].assign_weight(ctx, lt_name, tensor);
        return;
    }
    TTS_ASSERT(false);
}

void dac_quantize_layer::assign_weight(ggml_context * ctx, sv name, ggml_tensor * tensor) {
    static constexpr array<tuple<str, ggml_tensor * dac_quantize_layer::*, bool>, 3> OFFSETS{{
        {".codebook.weight", &dac_quantize_layer::codebook, false},
        {".out_proj.bias", &dac_quantize_layer::out_proj_bias, true},
        {".out_proj.weight", &dac_quantize_layer::out_proj_kernel, false},
    }};
    for (const auto [k, member, transpose] : OFFSETS) {
        if (k == name) {
            this->*member = ggml_dup_tensor(ctx, !transpose ? tensor : ggml_transpose(ctx, tensor));
            return;
        }
    }
    TTS_ASSERT(false);
}

void dac_model::assign_weight(sv name, ggml_tensor * weight) {
    static constexpr array<tuple<str, ggml_tensor * dac_model::*, bool>, 5> OFFSETS{{
        {"final.alpha", &dac_model::snake_alpha, true},
        {"final.bias", &dac_model::out_conv_bias, true},
        {"final.weight", &dac_model::out_conv_kernel, false},
        {"initial.bias", &dac_model::in_conv_bias, true},
        {"initial.weight", &dac_model::in_conv_kernel, false},
    }};
    if (const auto found{binary_search_idx(OFFSETS, name, pair_first_cmp_sv)}; ~found) {
        const auto [_, member, transpose] = OFFSETS[found];
        this->*member = copy_to_gpu(!transpose ? weight : ggml_transpose(&*ctx, weight));
        return;
    }
    if (const auto [index, lt_name] = parse_layer_count(name); ~index) {
        if (lt_name.contains("quantizers")) {
            quantizer_layers[index].assign_weight(&*ctx, lt_name, weight);
        } else {
            layers[index - 1].assign_weight(&*ctx, lt_name, weight);
        }
    }
}

static ggml_tensor * dac_build_audio_inputs(ggml_context * ctx, dac_context * dctx, const dac_ubatch & batch, vector<dac_quantize_layer> layers) {
    ggml_tensor * embd;
    
    dctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.sequence_length*dctx->model->n_heads);
    ggml_set_input(dctx->inp_tokens);

    if (dctx->backend) {
        ggml_backend_sched_set_tensor_backend(dctx->sched, dctx->inp_tokens, dctx->backend);
    }

    for(int i = 0; i < dctx->model->n_heads; i++) {
        dac_quantize_layer & quantize_layer = dctx->model->quantizer_layers[i];
        ggml_tensor * code = ggml_cont(ctx, ggml_view_2d(ctx, dctx->inp_tokens, 1, batch.sequence_length, dctx->model->n_heads*ggml_type_size(GGML_TYPE_I32), i*ggml_type_size(GGML_TYPE_I32)));
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

static ggml_tensor * build_residual_unit(ggml_context * ctx, ggml_tensor * cur, dac_residual_unit & u, int padding, int dilation) {
    ggml_tensor * residual = cur;
    cur = snake_1d(ctx, u.in_snake_alpha, cur);
    cur = ggml_conv_1d(ctx, u.in_conv_kernel, cur, 1, padding, dilation);
    cur = ggml_add(ctx, cur, u.in_conv_bias);
    cur = snake_1d(ctx, u.out_snake_alpha, cur);
    cur = ggml_conv_1d(ctx, u.out_conv_kernel,  cur, 1, 0, 1);
    cur = ggml_add(ctx, cur, u.out_conv_bias);
    return ggml_add(ctx, cur, residual);
}

static ggml_tensor * build_decoder_block(ggml_context * ctx, ggml_tensor * cur, dac_layer & l) {
    cur = snake_1d(ctx, l.snake_alpha_in, cur);
    cur = ggml_conv_transpose_1d(ctx, l.out_conv_kernel, cur, l.stride, l.padding, 1, 0, 1);
    cur = ggml_add(ctx, cur, l.out_conv_bias);
    for (int i = 0; i < l.residual_blocks.size(); i++) {
        cur = build_residual_unit(ctx, cur, l.residual_blocks[i], pow(3, i + 1), pow(3, i));
    }
    return cur;
}

dac_context::dac_context(dac_model * model, int n_threads, bool use_cpu) : runner_context{n_threads} {
}

void dac_runner::prepare_post_load() {
    dac_ubatch batch;
    batch.sequence_length = model->max_generation_size;
    ggml_cgraph * gf = build_dac_graph(batch);
    dctx->prep_schedule(gf);
}
    
ggml_cgraph * dac_runner::build_dac_graph(dac_ubatch & batch) {
    init_build();
    // splitting this out from the primary graph so that we can better manage streaming (i.e. sentence chunks are better performed this way)
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    
    ggml_tensor * cur;
    ggml_tensor * inputs;
    
    inputs = dac_build_audio_inputs(ctx, dctx, batch, model->quantizer_layers);
    ggml_set_name(inputs, "quantized_inputs");
    
    // everything besides the inputs is just a forward pass
    cur = ggml_conv_1d(ctx, model->in_conv_kernel, inputs, 1, 3, 1);
    cur = ggml_add(ctx, cur, model->in_conv_bias);
    for (auto l : model->layers) {
        cur = build_decoder_block(ctx, cur, l, dctx);
    }
    cur = snake_1d(ctx, model->snake_alpha, cur);
    cur = ggml_conv_1d(ctx, model->out_conv_kernel, cur, 1, 3, 1);
    cur = ggml_add(ctx, cur, model->out_conv_bias);
    cur = ggml_tanh(ctx, cur);
    ggml_build_forward_expand(gf, cur);
    free_build();
    return gf;
}

void dac_runner::run(uint32_t * input_tokens, uint32_t sequence_length, tts_response * outputs) {
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
        }

        dctx->buf_output = ggml_backend_buft_alloc_buffer(dctx->backend_cpu_buffer, new_size);
    }
    
    outputs->data = (float *) ggml_backend_buffer_get_base(dctx->buf_output);
    ggml_backend_buffer_clear(dctx->buf_output, 0);
    
    ggml_cgraph * gf = NULL;
    gf = build_dac_graph(batch);
    
    // the output is always the last tensor in the graph
    ggml_tensor * result = gf->nodes[gf->n_nodes - 1];
    ggml_backend_sched_alloc_graph(dctx->sched, gf);
    
    ggml_backend_tensor_set(dctx->inp_tokens, batch.input_tokens, 0, batch.sequence_length*model->n_heads*ggml_element_size(dctx->inp_tokens));

    ggml_backend_sched_graph_compute_async(dctx->sched, gf);

    dctx->get_ggml_node_data(result, outputs->data, batch.sequence_length*sizeof(float)*model->up_sampling_factor);

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(dctx->sched);
    outputs->n_outputs = sequence_length * model->up_sampling_factor;
}

tts_runner_with_dac::tts_runner_with_dac(float sampling_rate) : tts_runner_with_context{sampling_rate} {
    dac_model * audio_model = new dac_model{meta_ctx, weight_ctx, cpu_only};
    dac_context * dctx = build_new_dac_context(audio_model, n_threads, cpu_only);
    dac_runner * audio_decoder = new dac_runner(audio_model, dctx);
}
