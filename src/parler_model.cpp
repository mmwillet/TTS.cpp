#include "parler_model.h"

void parler_tts_model::prep_layers(gguf_context * meta_ctx) {
    layers.reserve((size_t) n_layers);
    for (int i = 0; i < (int) n_layers; i++) {
        parler_layer * l = new parler_layer{};
        layers.push_back(l);
    }
    
    embds.reserve((size_t) n_output_heads);
    heads.reserve((size_t) n_output_heads);
    for (int i = 0; i < n_output_heads; i++) {
        struct ggml_tensor * h = nullptr;
        struct ggml_tensor * embd = nullptr;
        embds.push_back(embd);
        heads.push_back(h);
    }
}

void parler_tts_model::prep_constants(gguf_context * meta) {
    // this is a bad pattern and I should fix it, but was being lazy
    int encode_length_key = gguf_find_key(meta, "encode_length");
    if (encode_length_key == -1) {
        TTS_ABORT("key 'encode_length' must be specified in gguf file.");
    }
    n_encode_length = gguf_get_val_u32(meta, encode_length_key);
    
    int hidden_size_key = gguf_find_key(meta, "hidden_size");
    if (hidden_size_key != -1) {
        hidden_size = gguf_get_val_u32(meta, hidden_size_key);
    }
    
    int output_heads_key = gguf_find_key(meta, "output_heads");
    if (output_heads_key != -1) {
        n_output_heads = gguf_get_val_u32(meta, output_heads_key);
    }
    
    int ctx_length_key = gguf_find_key(meta, "ctx_length");
    if (ctx_length_key != -1) {
        max_ctx_length = gguf_get_val_u32(meta, ctx_length_key);
    }
    
    int attn_heads_key = gguf_find_key(meta, "attn_heads");
    if (attn_heads_key != -1) {
        n_attn_heads = gguf_get_val_u32(meta, attn_heads_key);
        head_size = hidden_size / n_attn_heads;
    }
    
    int output_vocab_size_key = gguf_find_key(meta, "out_vocab_size");
    if (output_vocab_size_key != -1) {
        output_vocab_size = gguf_get_val_u32(meta, output_vocab_size_key);
    }
    
    int audio_vocab_size_key = gguf_find_key(meta, "audio_vocab_size");
    if (audio_vocab_size_key != -1) {
        audio_vocab_size = gguf_get_val_u32(meta, audio_vocab_size_key);
    }
    
    int max_gen_key = gguf_find_key(meta, "max_generation");
    if (max_gen_key != -1) {
        max_generation_size = gguf_get_val_u32(meta, max_gen_key);
    }
    
    int n_layers_key = gguf_find_key(meta, "num_hidden_layers");
    if (n_layers_key != -1) {
        n_layers = gguf_get_val_u32(meta, n_layers_key);
    }

    int max_cross_nodes_key = gguf_find_key(meta, "max_cross_nodes");
    if (max_cross_nodes_key != -1) {
        max_cross_nodes = gguf_get_val_u32(meta, max_cross_nodes_key);
    }

    int bos_token_id_key = gguf_find_key(meta, "bos_token_id");
    if (bos_token_id_key != -1) {
        bos_token_id = gguf_get_val_u32(meta, bos_token_id_key);
    }

    int eos_token_id_key = gguf_find_key(meta, "eos_token_id");
    if (eos_token_id_key != -1) {
        eos_token_id = gguf_get_val_u32(meta, eos_token_id_key);
    }
}

void parler_tts_model::prep_cross_key_values() {
    ggml_threadpool_t threadpool = nullptr;
    int n_threads = (int) get_cpu_count();
    ggml_backend_t backend_cpu = ggml_backend_cpu_init();
    ggml_backend_buffer_type_t backend_cpu_buffer = ggml_backend_cpu_buffer_type();
    ggml_backend_cpu_set_n_threads(backend_cpu, n_threads);
    ggml_backend_cpu_set_threadpool(backend_cpu, threadpool);
    std::vector<ggml_backend_buffer_type_t> bufs = {backend_cpu_buffer};
    std::vector<ggml_backend_t> backs = {backend_cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backs.data(), bufs.data(), 1, max_cross_nodes*n_layers, false);
    
    std::vector<uint8_t> buf_compute_meta;
    buf_compute_meta.resize(max_cross_nodes*n_layers*ggml_tensor_overhead() + ggml_graph_overhead_custom(max_cross_nodes*n_layers, false));
        
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * cctx = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(cctx, 4096, false);
    
    for (int i = 0; i < layers.size(); i++) {
        struct ggml_tensor * Kcur = ggml_mul_mat(cctx, layers[i]->attn_k_proj, precomputed_input_emb);
        struct ggml_tensor * Vcur = ggml_mul_mat(cctx, layers[i]->attn_v_proj, precomputed_input_emb);
        Kcur = ggml_reshape_3d(cctx, Kcur, head_size, n_attn_heads, n_encode_length);
        Vcur = ggml_transpose(cctx, Vcur);
        
        struct ggml_tensor * k = ggml_cont(cctx, ggml_permute(cctx, Kcur, 0, 2, 1, 3));
        ggml_set_name(k, ("cross_key_" + std::to_string(i)).c_str());
        ggml_build_forward_expand(gf, k);
        struct ggml_tensor * v = ggml_cont_3d(cctx, Vcur, n_encode_length, head_size, n_attn_heads);
        ggml_set_name(v, ("cross_value_" + std::to_string(i)).c_str());
        ggml_build_forward_expand(gf, v);
    }
    
    ggml_free(cctx);
    ggml_backend_sched_reserve(sched, gf);
    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_sched_graph_compute_async(sched, gf);
    
    for (int i = 0; i < layers.size(); i++) {
        struct ggml_tensor * k = ggml_graph_get_tensor(gf, ("cross_key_" + std::to_string(i)).c_str());
        layers[i]->cross_k = ggml_dup_tensor(ctx, k);
        set_tensor(layers[i]->cross_k, k);
        struct ggml_tensor * v = ggml_graph_get_tensor(gf, ("cross_value_" + std::to_string(i)).c_str());
        layers[i]->cross_v = ggml_dup_tensor(ctx, v);
        set_tensor(layers[i]->cross_v, v);
    }
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend_cpu);
}

void parler_tts_model::prep_buffers_and_context(bool cpu_only) {
    if (cpu_only) {
        backend = ggml_backend_cpu_init();
        buffer = ggml_backend_cpu_buffer_type();
    } else {
#ifdef GGML_METAL
        backend = ggml_backend_metal_init();
        buffer = ggml_backend_metal_buffer_type();
#endif
    }
    size_t ctx_size = ggml_tensor_overhead() * (size_t) (tensor_meta.n_tensors * 1.25);
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ctx = ggml_init(params);
    buf = ggml_backend_buft_alloc_buffer(buffer, tensor_meta.n_bytes + n_encode_length*hidden_size*sizeof(float)*n_layers*2);
    return;
}

void parler_tts_model::set_tensor(struct ggml_tensor * tensor, struct ggml_tensor * target) {
    tensor->buffer = buf;
    tensor->data = (void *)((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
    size_t size = ggml_nbytes(target);
    ggml_backend_tensor_set(tensor, target->data, 0, size);
    ggml_set_name(tensor, target->name);
    offset += size;
}

void parler_tts_model::setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only) {
    prep_constants(meta_ctx);
    prep_layers(meta_ctx);
    tensor_meta = compute_tensor_meta("decoder", load_context);
    prep_buffers_and_context(cpu_only);
}

size_t parler_tts_model::max_nodes() {
    return std::max<size_t>(8192, tensor_meta.n_tensors*5);
}

void parler_tts_model::free() {
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
