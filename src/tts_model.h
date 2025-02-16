#ifndef tts_model_h
#define tts_model_h

#include "util.h"

struct tts_model {
	struct model_tensor_meta tensor_meta;

    // this is the current byte offset into the model's buffer.
    size_t offset = 0;

    bool use_cross_attn = true;
    
    ggml_backend_buffer_type_t buffer = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    struct ggml_context * ctx;
    
	void prep_buffers_and_context(bool cpu_only, float size_offset, uint32_t dedicated_add_on_size);
    void setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only, std::string model_prefix, float size_offset = 1.4, uint32_t dedicated_add_on_size = 0);
    void set_tensor(struct ggml_tensor * tensor, struct ggml_tensor * target);
    size_t max_nodes();
    void free();
};

#endif
