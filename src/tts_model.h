#pragma once

#include <cstring>
#include <functional>
#include "util.h"
#include "common.h"

inline void copy_n(const ggml_tensor * first,
                   [[deprecated("use ggml_view_1d instead")]] size_t count,
                   vector<float> & result) {
    result.clear();
    result.resize(count);
    TTS_ASSERT(ggml_nbytes(first) <= count * sizeof(float));
    ggml_backend_tensor_get(first, result.data(), 0, count * sizeof(float));
}

void append_to_response(tts_response * response, tts_response * to_append);

struct ggml_threadpool_deleter { void operator()(ggml_threadpool_t threadpool) const { ggml_threadpool_free(threadpool); } };
typedef unique_ptr<ggml_threadpool, ggml_threadpool_deleter> ggml_threadpool_ptr;

class gpu_context {
    const ggml_backend_ptr cpu;
    // TODO: extend the backend and buffer support out to all devices
    const ggml_backend_ptr _gpu;
    const ggml_threadpool_ptr threadpool;
public:
    explicit gpu_context(int n_threads, bool cpu_only);
    ggml_backend * gpu() const { return _gpu ? &*_gpu : &*cpu; }
    ggml_backend_sched_ptr sched(size_t max_nodes) const;
};

/// Read/write hidden states and outputs
struct runner_context {
    explicit runner_context(const shared_ptr<gpu_context> & gpu, size_t max_nodes);
    virtual ~runner_context() = default;

    const shared_ptr<gpu_context> gpu;
    const ggml_backend_sched_ptr sched;
    const ggml_context_ptr ctx;
};

struct tts_runner_factory;

/// Readonly weights
struct tts_model {
    explicit tts_model(shared_ptr<gpu_context> gpu, const model_tensor_meta & tensor_meta);

    const ggml_context_ptr ctx;
    const ggml_backend_buffer_ptr buf;
    const size_t max_nodes;

protected:
    ggml_tensor * copy_to_gpu(ggml_tensor * src);
    virtual void assign_weights(const ggml_context & weights) = 0;

private:
    ggml_tallocr buf_offset;
};

struct tts_runner_with_context : tts_runner, runner_context {
    // TODO move to .cpp
    explicit tts_runner_with_context(int n_threads, size_t max_nodes, float sampling_rate = 44100.0f) : tts_runner{sampling_rate}, runner_context{n_threads, max_nodes} {};
};

struct tts_runner_with_dac : tts_runner_with_context {
    explicit tts_runner_with_dac(float sampling_rate = 44100.0f);
};
