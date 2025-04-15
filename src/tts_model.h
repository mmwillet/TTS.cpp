#ifndef tts_model_h
#define tts_model_h

#include "util.h"
#include "common.h"

using tensor_meta_callback = std::function<void(ggml_tensor*)>*;

struct runner_context {
	runner_context(int n_threads): n_threads(n_threads) {};
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
    
    void set_threads();
    void build_schedule(size_t max_nodes);
    bool prep_schedule(ggml_cgraph * gf);
};

struct tts_model {
	struct model_tensor_meta tensor_meta;

    // this is the current byte offset into the model's buffer.
    size_t offset = 0;

    bool use_cross_attn = true;
    
    ggml_backend_buffer_type_t buffer = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    tensor_meta_callback compute_tensor_meta_cb = nullptr;

    struct ggml_context * ctx;
    
	void prep_buffers_and_context(bool cpu_only, float size_offset, uint32_t dedicated_add_on_size);
    void setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only, std::string model_prefix, float size_offset = 1.4, uint32_t dedicated_add_on_size = 0);
    void set_tensor(struct ggml_tensor * tensor, struct ggml_tensor * target);
    size_t max_nodes();
    void assign_weight(std::string name, ggml_tensor * tensor);
    void free();
};

#endif
