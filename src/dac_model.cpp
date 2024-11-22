#include "dac_model.h"

void dac_model::prep_constants(gguf_context * meta) {
    // this is a bad pattern and I should fix it, but was being lazy
    int output_heads_key = gguf_find_key(meta, "output_heads");
    if (output_heads_key != -1) {
        n_heads = gguf_get_val_u32(meta, output_heads_key);;
    }

    int sampling_factor_key = gguf_find_key(meta, "up_sampling_factor");
    if (sampling_factor_key != -1) {
        up_sampling_factor = gguf_get_val_u32(meta, sampling_factor_key);
    }
    
    int max_gen_key = gguf_find_key(meta, "max_generation");
    if (max_gen_key != -1) {
        max_generation_size = gguf_get_val_u32(meta, max_gen_key);
    }
    
    for (int i = 0; i < (int) layers.size(); i++)  {
        int layer_stride_key = gguf_find_key(meta, ("dac_layer_stride_" + std::to_string(i)).c_str());
        if (layer_stride_key == -1) {
            TTS_ABORT("key %s must be specified in gguf file.", ("dac_layer_stride_" + std::to_string(i)).c_str());
        }
        layers[i].stride = gguf_get_val_u32(meta, layer_stride_key);
        int layer_padding_key = gguf_find_key(meta, ("dac_layer_padding_" + std::to_string(i)).c_str());
        if (layer_padding_key == -1) {
            TTS_ABORT("key %s must be specified in gguf file.", ("dac_layer_padding_" + std::to_string(i)).c_str());
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
        for (int ii = 0; ii < 3; ii++) {
            dac_residual_unit u;
            l.residual_blocks.push_back(u);
        }
        layers.push_back(l);
    }
}

void dac_model::prep_buffers_and_context(bool cpu_only, ggml_context * load_context) {
    backend = cpu_only ? ggml_backend_cpu_init() : ggml_backend_metal_init();
    // I suspect we don't need a buffer if we are going to be on a single device.
    buffer = cpu_only ? ggml_backend_cpu_buffer_type() : ggml_backend_metal_buffer_type();
    size_t ctx_size = ggml_tensor_overhead() * 5000; // * n_tensors;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ctx = ggml_init(params);
    buf = ggml_backend_buft_alloc_buffer(buffer, load_context->mem_size);//ggml_backend_cpu_buffer_from_ptr((char *) load_context->mem_buffer, load_context->mem_size);
}

void dac_model::set_tensor(struct ggml_tensor * tensor, struct ggml_tensor * target) {
    tensor->buffer = buf;
    tensor->data = (void *)((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
    size_t size = ggml_nbytes(target);
    ggml_backend_tensor_set(tensor, target->data, 0, size);
    offset += size;
}

void dac_model::setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only) {
    prep_layers(meta_ctx);
    prep_constants(meta_ctx);
    prep_buffers_and_context(cpu_only, load_context);
}

size_t dac_model::max_nodes() {
    return std::max<size_t>(8192, n_tensors*5);
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
