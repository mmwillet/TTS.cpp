#include "dia_model.h"
#include <ranges>
#include <regex>

dia_encoder::dia_encoder(size_t n_encoder_layers) : layers(n_encoder_layers) {}
dia_decoder::dia_decoder(size_t n_decoder_layers, size_t n_output_heads) :
        embds(n_output_heads), heads(n_output_heads), layers(n_decoder_layers) {}

void dia_model::assign_weight(sv name, ggml_tensor * tensor) {
    const auto dup{ggml_dup_tensor(ctx, tensor)};
    set_tensor(dup, tensor);

    const ptrdiff_t parts{ranges::count(name, '.')};
    TTS_ASSERT(parts >= 3);
    auto iter{(views::split(name, '.') | views::drop(1)).begin()};
    const sv encoder_or_decoder{*iter++};
    const sv embedding_norm_or_layers{*iter++};

    if (encoder_or_decoder == "encoder") {
        if (embedding_norm_or_layers == "embedding") {
            encoder.embedding = dup;
            return;
        }
        if (embedding_norm_or_layers == "norm") {
            encoder.norm = dup;
            return;
        }
        if (embedding_norm_or_layers == "layers") {
            TTS_ASSERT(parts >= 5);
            auto & layer = encoder.layers.at(sv_int(sv{*iter++}));
            static constexpr array<pair<str, ggml_tensor * dia_encoder_layer::*>, 9> OFFSETS{{
                {"gate", &dia_encoder_layer::gate},
                {"k_proj", &dia_encoder_layer::k},
                {"o_proj", &dia_encoder_layer::o},
                {"post_sa_norm", &dia_encoder_layer::mlp_norm},
                {"pre_sa_norm", &dia_encoder_layer::self_attn_norm},
                {"q_proj", &dia_encoder_layer::q},
                {"up", &dia_encoder_layer::up},
                {"v_proj", &dia_encoder_layer::v},
                {"wo", &dia_encoder_layer::out},
            }};
            static_assert(is_sorted(OFFSETS.begin(), OFFSETS.end()));
            if (const auto found{binary_search_idx(OFFSETS, sv{*iter}, pair_first_cmp_sv)}; ~found) {
                layer.*OFFSETS[found].second = dup;
                return;
            }
        }
    } else if (encoder_or_decoder == "decoder") {
        if (embedding_norm_or_layers == "norm") {
            decoder.norm = dup;
            return;
        }
        TTS_ASSERT(parts >= 4);
        const int index{sv_int(sv{*iter++})};
        if (embedding_norm_or_layers == "embeddings") {
            decoder.embds.at(index) = dup;
        } else if (embedding_norm_or_layers == "heads") {
            decoder.heads.at(index) = dup;
        } else if (embedding_norm_or_layers == "layers") {
            TTS_ASSERT(parts >= 5);
            auto & layer = decoder.layers.at(index);
            static constexpr array<pair<str, ggml_tensor * dia_decoder_layer::*>, 14> OFFSETS{{
                {"cross_k_proj", &dia_decoder_layer::cross_attn_k},
                {"cross_o_proj", &dia_decoder_layer::cross_attn_o},
                {"cross_q_proj", &dia_decoder_layer::cross_attn_q},
                {"cross_v_proj", &dia_decoder_layer::cross_attn_v},
                {"gate", &dia_decoder_layer::gate},
                {"pre_ca_norm", &dia_decoder_layer::cross_attn_norm},
                {"pre_mlp_norm", &dia_decoder_layer::mlp_norm},
                {"pre_sa_norm", &dia_decoder_layer::self_attn_norm},
                {"self_k_proj", &dia_decoder_layer::self_attn_k},
                {"self_o_proj", &dia_decoder_layer::self_attn_o},
                {"self_q_proj", &dia_decoder_layer::self_attn_q},
                {"self_v_proj", &dia_decoder_layer::self_attn_v},
                {"up", &dia_decoder_layer::up},
                {"wo", &dia_decoder_layer::out},
            }};
            static_assert(is_sorted(OFFSETS.begin(), OFFSETS.end()));
            if (const auto found{binary_search_idx(OFFSETS, sv{*iter}, pair_first_cmp_sv)}; ~found) {
                layer.*OFFSETS[found].second = dup;
                return;
            }
        }
    }

    TTS_ABORT("Unrecognized tensor '%s' when loading Dia from GGUF file.", name);
}

dia_model_constants::dia_model_constants(gguf_context * meta) {
    static constexpr array<pair<str, const uint32_t dia_model_constants::*>, 16> OFFSETS{{
        {"dia.attn_head_size", &dia_model_constants::head_size},
        {"dia.bos_token_id", &dia_model_constants::bos_token_id},
        {"dia.decoder.attn_heads", &dia_model_constants::decoder_attn_heads},
        {"dia.decoder.audio_vocab_size", &dia_model_constants::audio_vocab_size},
        {"dia.decoder.hidden_size", &dia_model_constants::decoder_hidden_size},
        {"dia.decoder.layers", &dia_model_constants::n_decoder_layers},
        {"dia.decoder.max_generation_size", &dia_model_constants::max_generation_size},
        {"dia.decoder.output_heads", &dia_model_constants::n_output_heads},
        {"dia.decoder.output_vocab_size", &dia_model_constants::output_vocab_size},
        {"dia.decoder.query_heads", &dia_model_constants::decoder_query_heads},
        {"dia.encoder.attn_heads", &dia_model_constants::encoder_attn_heads},
        {"dia.encoder.layers", &dia_model_constants::n_encoder_layers},
        {"dia.encoder.max_context_length", &dia_model_constants::max_encoder_context_length},
        {"dia.eos_token_id", &dia_model_constants::eos_token_id},
        {"dia.max_delay", &dia_model_constants::max_delay},
        {"dia.pad_token_id", &dia_model_constants::pad_token_id},
    }};
    static_assert(is_sorted(OFFSETS.begin(), OFFSETS.end()));
    for (const auto [i, key] : gguf_key_iterator{meta}) {
        if (const auto found{binary_search_idx(OFFSETS, sv{key}, pair_first_cmp_sv)}; ~found) {
            const_cast<uint32_t &>(this->*OFFSETS[found].second) = gguf_get_val_u32(meta, i);
        }
    }
    // please note that this value is not currently set in the gguf encoder as it effectively only exists as a default
    // python parameter (rather than an attribute in the model config) for the python Dia model.
    int cfg_scale_key = gguf_find_key(meta, "dia.cfg_scale");
    if (cfg_scale_key != -1) {
        const_cast<float &>(cfg_scale_data[0]) = gguf_get_val_f32(meta, cfg_scale_key);
    }
}

void dia_context::reset() {
    current_position = 0;
    prompt_size = 0;
    output_tokens.clear();
    delay_steps = -1;
}

dia_context::dia_context(dia_model * model_, int n_threads, bool use_cpu): runner_context{n_threads}, model{model_} {
    if (!use_cpu) {
#ifdef GGML_USE_METAL
        backend = ggml_backend_metal_init();
#endif
    }
    backend_cpu = ggml_backend_cpu_init();
    set_threads();
    build_schedule();
    buf_compute_meta.resize(ggml_tensor_overhead()*model->max_nodes() + ggml_graph_overhead_custom(model->max_nodes(), false));
}

dia_kv_cache_layer::dia_kv_cache_layer(ggml_context * ctx, int64_t gen_ne0, int64_t cross_ne0, int index) :
        k{ggml_new_tensor_1d(ctx, GGML_TYPE_F32, gen_ne0)}, v{ggml_new_tensor_1d(ctx, GGML_TYPE_F32, gen_ne0)},
        cross_k{ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cross_ne0)}, cross_v{ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cross_ne0)} {
    ggml_format_name(k, "cache_k_l%d", index);
    ggml_format_name(v, "cache_v_l%d", index);
    ggml_format_name(cross_k, "cache_cross_k_l%d", index);
    ggml_format_name(cross_v, "cache_cross_v_l%d", index);
}

dia_kv_cache::dia_kv_cache(const dia_model & model, bool is_cpu) : ctx{ggml_init({
            .mem_size   = (4u * model.n_decoder_layers + 1) * ggml_tensor_overhead(),
            .mem_buffer = NULL,
            .no_alloc   = true,
        })} {
    TTS_ASSERT(ctx);
    // this will only really support cpu or metal for the time being;
    buft = ggml_backend_cpu_buffer_type();
    if (!is_cpu) {
#ifdef GGML_USE_METAL
        buft = ggml_backend_metal_buffer_type();
#endif
    }

    layers.reserve(model.n_decoder_layers);
    const size_t gen_ne0{model.head_size * model.decoder_attn_heads * model.max_generation_size * 2};
    const size_t cross_ne0{model.head_size * model.decoder_attn_heads * model.max_encoder_context_length * 2};
    generate_n(back_inserter(layers), model.n_decoder_layers, [&]{ return {ctx, gen_ne0, cross_ne0, layers.size()}; });

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(&*ctx, buft));
    TTS_ASSERT(buf);
    ggml_backend_buffer_clear(&*buf, 0);
}


static ggml_tensor * dia_layer_norm(ggml_context * ctx, ggml_tensor * inputs, ggml_tensor * weight) {
    // dia always uses 1e-5 as the default eps
    constexpr float eps = 0.00001;
    return ggml_mul(ctx, ggml_rms_norm(ctx, inputs, eps), weight);
}

static inline void dia_residual_norm_lambda(ggml_context * ctx, ggml_tensor * & cur, ggml_tensor * norm_weight, auto lambda) {
    ggml_tensor * const residual{cur};
    cur = dia_layer_norm(ctx, cur, norm_weight);
    lambda();
    cur = ggml_add(ctx, cur, residual);
}

ggml_tensor * dia_model::build_dia_encoder(ggml_context * ctx, dia_context * dctx) {
    ggml_set_input(dctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_encoder_context_length*2));
    ggml_set_input(dctx->encode_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_encoder_context_length));
    ggml_set_input(dctx->encode_attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, max_encoder_context_length, max_encoder_context_length));

    ggml_tensor * cur = ggml_reshape_3d(ctx, ggml_get_rows(ctx, encoder.embedding, dctx->inp_tokens), encoder_hidden_size, max_encoder_context_length, 2);
    for (auto & layer : encoder.layers) {
        dia_residual_norm_lambda(ctx, cur, layer.self_attn_norm, [&]{ // self-attention
            ggml_tensor * Qcur = ggml_mul_mat(ctx, layer.q, cur);
            ggml_tensor * Kcur = ggml_mul_mat(ctx, layer.k, cur);
            ggml_tensor * Vcur = ggml_mul_mat(ctx, layer.v, cur);

            // Strangely Dia follows the neoX Rotary Positional Embeddings Protocol
            Qcur = ggml_rope(ctx, ggml_cont(ctx, ggml_reshape_4d(ctx, Qcur, head_size, encoder_attn_heads, max_encoder_context_length, 2)), dctx->encode_positions, head_size, 2);
            Kcur = ggml_rope(ctx, ggml_cont(ctx, ggml_reshape_4d(ctx, Kcur, head_size, encoder_attn_heads, max_encoder_context_length, 2)), dctx->encode_positions, head_size, 2);
            ggml_tensor * q = ggml_cont(ctx, ggml_permute(ctx, Qcur, 0, 2, 1, 3));
            ggml_tensor * k = ggml_cont(ctx, ggml_permute(ctx, Kcur, 0, 2, 1, 3));
            ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
            kq = ggml_soft_max_ext(ctx, kq, dctx->encode_attn_mask, 1.0f, 0.0f);
            ggml_tensor * v = ggml_cont_4d(ctx, ggml_transpose(ctx, Vcur), max_encoder_context_length, head_size, encoder_attn_heads, 2);
            ggml_tensor * kqv = ggml_mul_mat(ctx, kq, v);
            ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 2, 0, 1, 3);

            // It is unclear why the attention ops in Dia's encoder don't project to the embedding dimension size as is standard. Instead they up project to the decoder's embedding dimension
            // then down project back the the encoder embedding dimension.
            cur = ggml_cont_3d(ctx, kqv_merged, decoder_hidden_size, max_encoder_context_length, 2);
            cur = ggml_mul_mat(ctx, layer.o, cur);
        });
        dia_residual_norm_lambda(ctx, cur, layer.mlp_norm, [&]{ // mlp
            cur = ggml_mul(ctx, ggml_silu(ctx, ggml_mul_mat(ctx, layer.gate, cur)), ggml_mul_mat(ctx, layer.up, cur));
            cur = ggml_mul_mat(ctx, layer.out, cur);
        });
    }
    return dia_layer_norm(ctx, cur, encoder.norm);
}

static ggml_tensor * repeat_interleave_dim1(ggml_context * ctx, ggml_tensor * a, int repeat) {
    //return ggml_repeat(ctx, a, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a->ne[0], 4*a->ne[1], a->ne[2], a->ne[3]));
    ggml_tensor * running{};
    for (int i = 0; i < a->ne[1]; i++) {
        int offset = i * a->nb[1];
        ggml_tensor * t = ggml_cont(ctx, ggml_view_4d(ctx, a, a->ne[0], 1, a->ne[2], a->ne[3], a->nb[1], a->nb[2], a->nb[3], offset));
        t = ggml_repeat(ctx, t, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a->ne[0], repeat, a->ne[2], a->ne[3]));
        running = !i ? t : ggml_concat(ctx, running, t, 1);
    }
    return running;
}

void dia_model::build_dia_self_kv_store(ggml_context * ctx, dia_context * dctx, dia_kv_cache_layer & kv, ggml_cgraph * gf, ggml_tensor * k, ggml_tensor * v, dia_ubatch & batch) {
    const size_t attn_size{head_size * decoder_attn_heads};
    {
        k = ggml_rope(ctx, ggml_cont(ctx, ggml_reshape_4d(ctx, k, head_size, decoder_attn_heads / decoder_query_heads, batch.sequence_length, 2)), dctx->positions, head_size, 2);
        // Since the sequence length should always be 1 here this is the most pertinent time to repeat the heads for grouped query attention.
        // If GGML supported a repeat_interleave op then it would be more optimal to store just the groups in the cache and interleave the attention heads after recalling
        // from the cache
        k = repeat_interleave_dim1(ctx, ggml_cont(ctx, ggml_reshape_4d(ctx, k, head_size, decoder_attn_heads / decoder_query_heads, batch.sequence_length, 2)), decoder_query_heads);
        k = ggml_cont(ctx, ggml_reshape_2d(ctx, k, attn_size, 2));

        const size_t attn_size_k{attn_size * ggml_element_size(kv.k)};
        ggml_build_forward_expand(gf, ggml_cpy(ctx, k, ggml_view_2d(ctx, kv.k, attn_size, 2, attn_size_k * max_generation_size, attn_size_k * dctx->current_position)));
    }
    {
        // Since the sequence length should always be 1 here this is the most pertinent time to repeat the heads for grouped query attention.
        // If GGML supported a repeat_interleave op then it would be more optimal to store just the groups in the cache and interleave the attention heads after recalling
        // from the cache
        v = repeat_interleave_dim1(ctx, ggml_cont(ctx, ggml_reshape_4d(ctx, v, head_size, decoder_attn_heads / decoder_query_heads, batch.sequence_length, 2)), decoder_query_heads);

        const size_t attn_size_v{attn_size * ggml_element_size(kv.v)};
        ggml_build_forward_expand(gf, ggml_cpy(ctx, v, ggml_view_2d(ctx, kv.v, attn_size, 2, attn_size_v * max_generation_size, attn_size_v * dctx->current_position)));
    }
}

void dia_model::build_dia_cross_kv_store(ggml_context * ctx, dia_context * dctx, dia_kv_cache_layer & kv, ggml_cgraph * gf, dia_decoder_layer & layer, ggml_tensor * encoder_hidden_states) {
    {
        ggml_tensor * const positions_view{ggml_view_1d(ctx, dctx->encode_positions, dctx->prompt_size, 0)};
        const size_t sne0{encoder_hidden_size * ggml_element_size(encoder_hidden_states)}, sne1{sne0 * max_encoder_context_length};
        ggml_tensor * const encoder_states_key_view{ggml_cont(ctx, ggml_view_3d(ctx, encoder_hidden_states, encoder_hidden_size, dctx->prompt_size, 2, sne0, sne1, 0))};

        ggml_tensor * k = ggml_mul_mat(ctx, layer.cross_attn_k, encoder_states_key_view);
        k = ggml_rope(ctx, ggml_cont(ctx, ggml_reshape_4d(ctx, k, head_size, decoder_attn_heads, dctx->prompt_size, 2)), positions_view, head_size, 2);
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 1, 3, 2));

        const size_t kne0{head_size * ggml_element_size(kv.cross_k)}, kne1{kne0 * decoder_attn_heads}, kne2{kne1 * 2};
        ggml_build_forward_expand(gf, ggml_cpy(ctx, k, ggml_view_4d(ctx, kv.cross_k, head_size, decoder_attn_heads, 2, dctx->prompt_size, kne0, kne1, kne2, 0)));
    }
    {
        ggml_tensor * v = ggml_cont(ctx, ggml_transpose(ctx, ggml_mul_mat(ctx, layer.cross_attn_v, encoder_hidden_states)));
        v = ggml_cont_4d(ctx, v, max_encoder_context_length, head_size, decoder_attn_heads, 2);

        const size_t vne0{max_encoder_context_length * ggml_element_size(kv.cross_v)}, vne1{vne0 * head_size}, vne2{vne1 * decoder_attn_heads};
        ggml_build_forward_expand(gf, ggml_cpy(ctx, v, ggml_view_4d(ctx, kv.cross_v, max_encoder_context_length, head_size, decoder_attn_heads, 2, vne0, vne1, vne2, 0)));
    }
}

ggml_tensor * dia_model::build_dia_decoder(
        ggml_cgraph * gf,
        ggml_context * ctx,
        dia_context * dctx,
        dia_kv_cache & cache,
        dia_ubatch & batch,
        ggml_tensor * encoder_hidden_states) {
    ggml_set_input(dctx->positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.sequence_length));
    ggml_tensor * cur{};
    { // build_dia_decoder_inp_embd
        ggml_set_input(dctx->audio_inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_output_heads * 2));
        for (const auto embd : decoder.embds) {
            ggml_tensor * const view{ggml_view_1d(ctx, dctx->audio_inp_tokens, 2, i * ggml_element_size(dctx->audio_inp_tokens))};
            view->nb[0] = n_output_heads * ggml_element_size(dctx->audio_inp_tokens);
            ggml_tensor * const addend{ggml_get_rows(ctx, embd, view)};
            cur = !cur ? addend : ggml_add(ctx, addend, cur);
        }
    }
    for (auto [layer, kv] : views::zip(decoder.layers, cache.layers)) {
        dia_residual_norm_lambda(ctx, cur, layer.self_attn_norm, [&]{ // self-attention
            ggml_tensor * const Kcur{ggml_mul_mat(ctx, layer.self_attn_k, cur)};
            ggml_tensor * const Vcur{ggml_mul_mat(ctx, layer.self_attn_v, cur)};
            build_dia_self_kv_store(ctx, dctx, kv, gf, Kcur, Vcur, batch);

            const size_t kne0{head_size * ggml_element_size(kv.k)}, kne1{kne0 * decoder_attn_heads}, kne2{kne1 * max_generation_size};
            ggml_tensor * k = ggml_view_4d(ctx, kv.k, head_size, decoder_attn_heads, dctx->current_position + 1, 2, kne0, kne1, kne2, 0);
            k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));

            const size_t vne0{decoder_attn_heads * head_size * ggml_element_size(kv.v)}, vne1{vne0 * max_generation_size};
            ggml_tensor * v = ggml_view_3d(ctx, kv.v, head_size * decoder_attn_heads, dctx->current_position + 1, 2, vne0, vne1, 0);
            v = ggml_cont_4d(ctx, ggml_transpose(ctx, v), dctx->current_position + 1, head_size, decoder_attn_heads, 2);

            // As noted in the encoder Dia uses the Neo-X protocol for RoPE.
            ggml_tensor * Qcur = ggml_mul_mat(ctx, layer.self_attn_q, cur);
            Qcur = ggml_rope(ctx, ggml_cont(ctx, ggml_reshape_4d(ctx, Qcur, head_size, decoder_attn_heads, batch.sequence_length, 2)), dctx->positions, head_size, 2);
            ggml_tensor * q = ggml_cont(ctx, ggml_permute(ctx, Qcur, 0, 2, 1, 3));
            ggml_tensor * kq = ggml_mul_mat(ctx, ggml_cont(ctx, k), q);

            // given that attention bias, scaling and masking are not used for decoding, it might be faster to prefer the #ggml_soft_max op here,
            kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f, 0.0f);
            ggml_tensor * kqv = ggml_mul_mat(ctx, kq, v);
            ggml_tensor * kqv_merged = ggml_cont(ctx, ggml_permute(ctx, kqv, 2, 0, 1, 3));
            cur = ggml_cont_3d(ctx, kqv_merged, decoder_hidden_size, batch.sequence_length, 2);
            cur = ggml_mul_mat(ctx, layer.self_attn_o, cur);

            // if we ever need to support multiple step decoder runs then this reshape will need to be replaced with permutation.
            cur = ggml_cont_2d(ctx, cur, cur->ne[0], 2);
        });
        dia_residual_norm_lambda(ctx, cur, layer.cross_attn_norm, [&]{ // cross-attention
            // only load the cross attention kv store when performing the encoding step
            if (batch.encoder_step) {
                build_dia_cross_kv_store(ctx, dctx, kv, gf, layer, encoder_hidden_states);
            }

            const size_t kne0{head_size * ggml_element_size(kv.cross_k)}, kne1{kne0 * decoder_attn_heads}, kne2{kne1 * 2};
            ggml_tensor * cross_k = ggml_view_4d(ctx, kv.cross_k, head_size, decoder_attn_heads, 2, max_encoder_context_length, kne0, kne1, kne2, 0);
            // the double permute operation shouldn't be necessary here, but it seems that currently ggml permute only currently alows for a single
            // axis pair to be transposed.
            cross_k = ggml_cont(ctx, ggml_permute(ctx, ggml_permute(ctx, cross_k, 0, 1, 3, 2), 0, 2, 1, 3));

            const size_t vne0{max_encoder_context_length * ggml_element_size(kv.cross_v)}, vne1{vne0 * head_size}, vne2{vne1 * decoder_attn_heads};
            ggml_tensor * cross_v = ggml_cont(ctx, ggml_view_4d(ctx, kv.cross_v, max_encoder_context_length, head_size, decoder_attn_heads, 2, vne0, vne1, vne2, 0));

            // As noted in the encoder Dia uses the Neo-X protocol for RoPE.
            ggml_tensor * cross_Qcur = ggml_mul_mat(ctx, layer.cross_attn_q, cur);
            cross_Qcur = ggml_rope(ctx, ggml_cont(ctx, ggml_reshape_4d(ctx, cross_Qcur, head_size, decoder_attn_heads, batch.sequence_length, 2)), dctx->positions, head_size, 2);
            ggml_tensor * cross_q = ggml_cont(ctx, ggml_permute(ctx, cross_Qcur, 0, 2, 1, 3));

            ggml_tensor * cross_kq = ggml_mul_mat(ctx, cross_k, cross_q);
            // given that attention bias, scaling and masking are not used for decoding, it might be faster to prefer the #ggml_soft_max op here,
            cross_kq = ggml_soft_max_ext(ctx, cross_kq, nullptr, 1.0f, 0.0f);
            ggml_tensor * cross_kqv = ggml_mul_mat(ctx, cross_kq, cross_v);
            ggml_tensor * cross_kqv_merged = ggml_cont(ctx, ggml_permute(ctx, cross_kqv, 2, 0, 1, 3));
            cur = ggml_cont_3d(ctx, cross_kqv_merged, decoder_hidden_size, batch.sequence_length, 2);
            cur = ggml_mul_mat(ctx, layer.cross_attn_o, cur);

            // if we ever need to support multiple step decoder runs then this reshape will need to be replaced with permutation.
            cur = ggml_cont_2d(ctx, cur, cur->ne[0], 2);
        });
        dia_residual_norm_lambda(ctx, cur, layer.mlp_norm, [&]{ // mlp
            cur = ggml_mul(ctx, ggml_silu(ctx, ggml_mul_mat(ctx, layer.gate, cur)), ggml_mul_mat(ctx, layer.up, cur));
            cur = ggml_mul_mat(ctx, layer.out, cur);
        });
    }
    { // build_dia_head_outputs
        cur = dia_layer_norm(ctx, cur, decoder.norm);
        // going to cat the heads together and then reshape them
        ggml_tensor * out{};
        for (const auto head : decoder.heads) {
            ggml_tensor * const concatenand = ggml_mul_mat(ctx, head, cur);
            out = !out ? concatenand : ggml_concat(ctx, out, concatenand, 2);
        }
        ggml_tensor * const cond = ggml_cont(ctx, ggml_view_2d(ctx, out, out->ne[0], out->ne[2], out->nb[2], 0));
        ggml_tensor * const uncond = ggml_cont(ctx, ggml_view_2d(ctx, out, out->ne[0], out->ne[2], out->nb[2], out->nb[1]));
        cur = ggml_map_custom2(ctx, cond, uncond, &cfg_scale, out->ne[0], const_cast<float*>(cfg_scale_data));
    }
    return cur;
}

void dia_runner::tokenize_sentence(str sentence_, dia_ubatch & batch) {
    // Dia's tokenization process is unusual. Essentially Dia takes the byte value for each character and uses that as
    // a token array. Additionally, because Dia performs a cfg-scale adjustment before sampling tokens, it is necessary to
    // generate with a conditioned context (i.e. with the text) and an unconditioned context (i.e. without any text) so that
    // proper adjustments can be perfored at each generation step. This means that we need to pad the end of our tokens to the
    // max context size for both the conditional and unconditional sequence.

    // if the sentence isn't prepended by dialogue start tokens, [S1] or [S2], then append one.
    string sentence{strip(sentence_)};
    if (!sentence.starts_with("[S1]") || !sentence.starts_with("[S2]")) {
        sentence = "[S1] " + sentence;
    }
    if (sentence.back() != '.') {
        sentence += ".";
    }

    // [S1] and [S2] are special character sequences that are replaced with the special tokens 0x01 and 0x02 respectively.
    sentence = regex_replace(sentence, regex{"\\[S1\\]"}, "\x01");
    sentence = regex_replace(sentence, regex{"\\[S2\\]"}, "\x02");

    if (sentence.size() > model->max_encoder_context_length) {
        TTS_ABORT("Dia currently only supports a max of %d characters and received an input of %d characters.", model->max_encoder_context_length, sentence.size());
    }
    batch.tokens.reserve(model->max_encoder_context_length * 2);
    batch.tokens.assign(cbegin(sentence), cend(sentence));
    batch.sentence_length = batch.tokens.size();
    // this 100 token warning is arbitrarily chosen based on spot checking small prompt performance
    if (batch.sentence_length <= 100) {
        fprintf(stdout, "Your prompt has fewer than 100 tokens. Please note that Dia's generation with prompts that are fewer than 100 tokens is highly inconsistent.\n");
    }

    batch.tokens.resize(model->max_encoder_context_length * 2);
 }

dia_ubatch dia_runner::batch_from_sentence(str sentence) {
    // if we are generating a new batch from tokens then we need to run the encoder step;
    dia_ubatch batch{1, true};
    tokenize_sentence(sentence, batch);
    batch.audio_tokens.resize(model->n_output_heads, model->bos_token_id);
    return batch;
}

/*
 * There are two unique features of Dia's model architecture:
 * 1.  Dia cleans its output generation by adding the difference between its text based output (its conditional output) and its unconditional output
 *     to the conditional ouput before sampling. This is why the batch is set to two throughout the graph.
 *
 * 2.  Dia's decoder attends across the entire encoded space including the pad buffer which receives a unique attention mask. This is why the
 *     encoder sequence is always max length.
 */
ggml_cgraph * dia_runner::build_dia_graph(dia_ubatch & batch) {
    init_build();
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_tensor * encoded_states{};

    if (batch.encoder_step) {
        encoded_states = model.build_dia_encoder(ctx, dctx, batch);
        ggml_build_forward_expand(gf, encoded_states);
    }

    ggml_tensor * cur = build_dia_decoder(gf, ctx, model, dctx, kv_cross_self, batch, encoded_states);
    ggml_set_name(cur, "decoder_output");
    ggml_build_forward_expand(gf, cur);
    free_build();

    return gf;
}

void dia_runner::configure_generation(const generation_configuration & config) {
    GGML_ASSERT(config.max_tokens == 0 || config.max_tokens > model->max_delay);
    decode_sampler->temperature = config.temperature;
    decode_sampler->repetition_penalty = config.repetition_penalty;
    decode_sampler->do_sample = config.sample;
    decode_sampler->top_k = config.top_k;
    dctx->max_generation_size = config.max_tokens > model->max_delay ? config.max_tokens : model->max_generation_size;
}

void dia_runner::set_inputs(dia_ubatch & batch) {
    if (batch.encoder_step) {
        ggml_backend_tensor_set(dctx->inp_tokens, batch.tokens.data(), 0, batch.tokens.size()*ggml_element_size(dctx->inp_tokens));
        int32_t * const ep{dctx->encode_positions->data};
        float * const mask{dctx->encode_attn_mask->data};
        const size_t i_max{model->max_encoder_context_length};
        const size_t n{batch.sentence_length};
        TTS_ASSERT(n < i_max);
        iota(ep, ep + i_max, 0);
        fill_n(mask, i_max * i_max, 0);
        for (int i = 0; i < n; i++) {
            fill_n(mask + i*i_max + n, i_max - n, -INFINITY);
        }
        for (int i = n; i < i_max; i++) {
            fill_n(mask + i*i_max, n, -INFINITY);
        }
    }
    // The audio tokens need to be repeated in the input in order to support cfg-scaling. I.E we need duplicate inputs for conditional and unconditional logits.
    const size_t len{batch.audio_tokens.size()*ggml_element_size(dctx->audio_inp_tokens)};
    ggml_backend_tensor_set(dctx->audio_inp_tokens, batch.audio_tokens.data(), 0, len);
    ggml_backend_tensor_set(dctx->audio_inp_tokens, batch.audio_tokens.data(), len, len);
    ((int32_t*) dctx->positions->data)[0] = dctx->current_position;
}

void dia_runner::decode(dia_ubatch & batch) {
    if (batch.encoder_step) {
        dctx->prompt_size = batch.sentence_length;
        dctx->output_tokens.reserve(dctx->max_generation_size * model->n_output_heads);
    }
    ggml_backend_sched_reset(dctx->sched);

    const size_t logits_size = model->output_vocab_size * dctx->max_generation_size * model->n_output_heads;
    const size_t prev_size = dctx->buf_output ? ggml_backend_buffer_get_size(dctx->buf_output) : 0;
    const size_t new_size  = logits_size * sizeof(float);

    if (!dctx->buf_output || prev_size < new_size) {
        if (dctx->buf_output) {
            ggml_backend_buffer_free(dctx->buf_output);
            dctx->buf_output = nullptr;
            dctx->logits = nullptr;
        }

        dctx->buf_output = ggml_backend_buft_alloc_buffer(dctx->backend_cpu_buffer, new_size);
    }

    dctx->logits = (float *) ggml_backend_buffer_get_base(dctx->buf_output);

    ggml_cgraph * gf = build_dia_graph(batch);

    // the output is always the last tensor in the graph
    ggml_tensor * res = gf->nodes[gf->n_nodes - 1];
    ggml_backend_sched_alloc_graph(dctx->sched, gf);

    set_inputs(batch);

    ggml_backend_sched_graph_compute_async(dctx->sched, gf);

    float * logits_out = dctx->logits + dctx->current_position * model->output_vocab_size * model->n_output_heads;
    dctx->get_ggml_node_data(res, logits_out, model->output_vocab_size * model->n_output_heads * sizeof(float));

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(dctx->sched);
}

dia_ubatch dia_runner::build_worst_case_batch()  {
    dia_ubatch batch{1, true};
    batch.tokens.resize(model->max_encoder_context_length * 2);
    batch.audio_tokens.resize(model->n_output_heads);
    return batch;
}

void dia_runner::prepare_post_load() {
    dac_runner->prepare_post_load();
    dia_kv_cache_init(kv_cross_self, model, dctx);
    dia_ubatch batch = build_worst_case_batch();
    batch.sentence_length = model->max_encoder_context_length;
    dctx->prompt_size = model->max_encoder_context_length;
    ggml_cgraph * gf = build_dia_graph(batch);
    dctx->prep_schedule(gf);
}

static constexpr array<uint32_t> DELAY_PATTERN{0, 8, 9, 10, 11, 12, 13, 14, 15};

bool dia_runner::check_stopping(dia_ubatch & batch) {
    if (dctx->delay_steps == -1 && (batch.audio_tokens[0] == model->eos_token_id || dctx->current_position >= dctx->max_generation_size - model->max_delay)) {
        dctx->delay_steps = model->max_delay;
    }
    if (dctx->delay_steps > 0) {
        const int step_after_eos{model->max_delay - dctx->delay_steps};
        for (const auto & [delay, audio_token] : views::zip{DELAY_PATTERN, batch.audio_tokens}) {
            if (delay < step_after_eos) {
                audio_token = delay == step_after_eos ? model->eos_token_id : model->pad_token_id;
            }
        }
        dctx->delay_steps -= 1;
    }
    return !dctx->delay_steps;
}

vector<uint32_t> dia_runner::adjust_output_tokens(const vector<uint32_t> & output_tokens) {
    // currently this is applying sliding window over the heads and filtering out bad tokens.
    // If we convert the DAC model's quantizer layers to support by row + column embeddings then we will need to transpose
    // the heads and the sequence here, but right now simplying using a strided view is more peformant.
    TTS_ASSERT(model->n_output_heads <= DELAY_PATTERN.size());
    const size_t size{output_tokens.size()};
    vector<uint32_t> filtered{};
    filtered.reserve(size);
    for (int i = 0; i < (size / model->n_output_heads) - model->max_delay; i++) {
        const size_t rewind_size{filtered.size()};
        for (int ii = 0; ii < model->n_output_heads; ii++) {
            const int next_index{i*model->n_output_heads+model->DELAY_PATTERN[ii]*model->n_output_heads+ii};
            if (next_index > size || output_tokens[next_index] >= model->audio_vocab_size) {
                filtered.resize(rewind_size);
                break;
            }
            filtered.push_back(output_tokens[next_index]);
        }
    }
    return filtered;
}

int dia_runner::generate_from_batch(dia_ubatch & batch, tts_response * output) {
    while (!check_stopping(batch)) {
        if (const int state{decode(batch)}) {
            return state;
        }
        decode_sampler->sample(dctx->logits + dctx->current_position * model->n_output_heads * model->output_vocab_size, dctx->output_tokens);
        dctx->current_position += batch.sequence_length;
        batch = dia_ubatch{ 1 };
        uint32_t * const last_outputs{ctx->output_tokens.data() + (int) dctx->output_tokens.size() - model->n_output_heads};
        batch.audio_tokens.resize(model->n_output_heads, model->bos_token_id);
        TTS_ASSERT(dctx->current_position <= model->n_output_heads);
        batch.audio_tokens.assign(last_outputs, last_outputs + dctx->current_position);
    }

    const vector<uint32_t> filtered_output_tokens{adjust_output_tokens(dctx->output_tokens, filtered_output_tokens)};
    dac_runner->run(filtered_output_tokens.data(), (int32_t) filtered_output_tokens.size() / model->n_output_heads, output);
    return 0;
}

int dia_runner::generate(str sentence, tts_response * output) {
    dia_ubatch batch = batch_from_sentence(sentence);
    dctx->reset();
    decode_sampler->reset();
    dctx->current_position = 0;
    if (!kv_cross_self) {
        kv_cross_self = new dia_kv_cache(kv_cross_self, model, dctx));
    }
    return generate_from_batch(batch, output);
}

void dia_runner::assign_weight(sv name, ggml_tensor * tensor) {
    if (!tensor->data) {
        return;
    }

    if (name.empty()) {
        // handles the top level meta tensor
        return;
    }

    if (name.starts_with("audio_encoder.")) {
        dac_runner->model->assign_weight(name + 14, tensor);
    } else {
        model->assign_weight(name, tensor);
    }
}
