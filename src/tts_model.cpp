#include "tts_model.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

void runner_context::set_threads() {
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

void runner_context::build_schedule(size_t max_nodes) {
    backend_cpu_buffer = ggml_backend_cpu_buffer_type();
    if (backend != nullptr) {
#ifdef GGML_USE_METAL
        backend_buffer = ggml_backend_metal_buffer_type();
#endif
        std::vector<ggml_backend_buffer_type_t> bufs = {backend_buffer, backend_cpu_buffer};
        std::vector<ggml_backend_t> backs = {backend, backend_cpu};
        sched = ggml_backend_sched_new(backs.data(), bufs.data(), 2, max_nodes, false);
    } else {
        std::vector<ggml_backend_buffer_type_t> bufs = {backend_cpu_buffer};
        std::vector<ggml_backend_t> backs = {backend_cpu};
        sched = ggml_backend_sched_new(backs.data(), bufs.data(), 1, max_nodes, false);
    }
}

bool runner_context::prep_schedule(struct ggml_cgraph * gf) {
    return ggml_backend_sched_reserve(sched, gf);
}

void tts_runner::init_build(std::vector<uint8_t>* buf_compute_meta) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta->size(),
        /*.mem_buffer =*/ buf_compute_meta->data(),
        /*.no_alloc   =*/ true,
    };

    ctx = ggml_init(params);
}
    
void tts_runner::free_build() {
    if (ctx) {
        ggml_free(ctx);
        ctx = nullptr;
    }
}

void tts_model::prep_buffers_and_context(bool cpu_only, float size_offset, uint32_t dedicated_add_on_size) {
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
    size_t ctx_size = ggml_tensor_overhead() * (tensor_meta.n_tensors * size_offset);
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ctx = ggml_init(params);
    buf = ggml_backend_buft_alloc_buffer(buffer, tensor_meta.n_bytes + dedicated_add_on_size);
}

void tts_model::assign_weight(std::string name, ggml_tensor * tensor) {
	TTS_ABORT("%s received name, %s, tensor without being defined. %s must be defined for all implementations of tts_model. \n", __func__, name.c_str(), __func__);
}

void tts_model::set_tensor(struct ggml_tensor * tensor, struct ggml_tensor * target) {
    tensor->buffer = buf;
    tensor->data = (void *)((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
    size_t size = ggml_nbytes(target);
    ggml_backend_tensor_set(tensor, target->data, 0, size);
    ggml_set_name(tensor, target->name);
    offset += size;
}

void tts_model::setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only, std::string model_prefix, float size_offset, uint32_t dedicated_add_on_size) {
    tensor_meta = compute_tensor_meta(model_prefix, load_context, compute_tensor_meta_cb);
    prep_buffers_and_context(cpu_only, size_offset, dedicated_add_on_size);
}

size_t tts_model::max_nodes() {
    return std::max<size_t>(8192, tensor_meta.n_tensors*5);
}

void tts_model::free() {
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
