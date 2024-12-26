#include "dac_model.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

void dac_model::prep_constants(gguf_context * meta) {
    int output_heads_key = search_for_gguf_keys(meta, {"parler-tts.decoder.output_heads", "output_heads"});
    if (output_heads_key != -1) {
        n_heads = gguf_get_val_u32(meta, output_heads_key);;
    }

    int sampling_factor_key = search_for_gguf_keys(meta, {"dac.up_sampling_factor", "up_sampling_factor"});
    if (sampling_factor_key != -1) {
        up_sampling_factor = gguf_get_val_u32(meta, sampling_factor_key);
    }
    
    int max_gen_key = search_for_gguf_keys(meta, {"parler-tts.decoder.max_generation", "max_generation"});
    if (max_gen_key != -1) {
        max_generation_size = gguf_get_val_u32(meta, max_gen_key);
    }
    
    for (int i = 0; i < (int) layers.size(); i++)  {
        std::string stride_kw = "dac_layer_stride_" + std::to_string(i);
        std::string padding_kw = "dac_layer_padding_" + std::to_string(i);
        int layer_stride_key = search_for_gguf_keys(meta, {"dac." + stride_kw, stride_kw});
        if (layer_stride_key == -1) {
            TTS_ABORT("key %s must be specified in gguf file.", ("dac." + stride_kw).c_str());
        }
        layers[i].stride = gguf_get_val_u32(meta, layer_stride_key);
        int layer_padding_key = search_for_gguf_keys(meta, {"dac." + padding_kw, padding_kw});
        if (layer_padding_key == -1) {
            TTS_ABORT("key %s must be specified in gguf file.", ("dac." + padding_kw).c_str());
        }
        layers[i].padding = gguf_get_val_u32(meta, layer_padding_key);
    }
}

void dac_model::prep_layers(gguf_context * meta) {
    for (int i = 0; i < n_heads; i++) {
        dac_quantize_layer l;
        quantizer_layers.push_back(l);
    }
    
    for (int i = 0; i < n_layers; i++) {
        dac_layer l;
        // all dac layers have 3 residual units
        for (int ii = 0; ii < 3; ii++) {
            dac_residual_unit u;
            l.residual_blocks.push_back(u);
        }
        layers.push_back(l);
    }
}

void dac_model::prep_buffers_and_context(bool cpu_only) {
    // currently DAC is only supported on cpu because the ops are not implemented on other devices;
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
    size_t ctx_size = ggml_tensor_overhead() * (tensor_meta.n_tensors * 1.25);
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ctx = ggml_init(params);
    buf = ggml_backend_buft_alloc_buffer(buffer, tensor_meta.n_bytes);
}

void dac_model::set_tensor(struct ggml_tensor * tensor, struct ggml_tensor * target) {
    tensor->buffer = buf;
    tensor->data = (void *)((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
    size_t size = ggml_nbytes(target);
    ggml_backend_tensor_set(tensor, target->data, 0, size);
    ggml_set_name(tensor, target->name);
    offset += size;
}

void dac_model::setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only) {
    prep_layers(meta_ctx);
    prep_constants(meta_ctx);
    tensor_meta = compute_tensor_meta("audio_encoder", load_context);
    prep_buffers_and_context(cpu_only);
}

size_t dac_model::max_nodes() {
    return std::max<size_t>(8192, tensor_meta.n_tensors*5);
}

void dac_model::free() {
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
