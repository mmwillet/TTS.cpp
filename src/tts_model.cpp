#include "tts_model.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

void append_to_response(tts_response * response, tts_response * to_append) {
    // TODO fix memory leak
    float * new_data = (float *) malloc((response->n_outputs + to_append->n_outputs) * sizeof(float));
    memcpy(new_data, response->data, response->n_outputs*sizeof(float));
    float * next_loc = new_data + response->n_outputs;
    memcpy(next_loc, to_append->data, to_append->n_outputs*sizeof(float));
    response->data = new_data;
    response->n_outputs += to_append->n_outputs;
}

gpu_context::gpu_context(int n_threads, bool cpu_only)
    : cpu{ggml_backend_cpu_init()}, _gpu{
        cpu_only ? nullptr : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr)
      }, threadpool{
          [n_threads] {
              ggml_threadpool_params ttp{ggml_threadpool_params_default(n_threads)};
              return ggml_threadpool_new(&ttp);
          }()
      } {
    ggml_backend_cpu_set_n_threads(&*cpu, n_threads);
    ggml_backend_cpu_set_threadpool(&*cpu, &*threadpool);
    if (!cpu_only) {
        if (!_gpu) {
            TTS_ABORT("'GGML_USE_METAL' is not defined either set the model to use CPU only or install ggml with metal support.");
        }
#ifdef GGML_USE_METAL
        // this is form copied from llama.cpp, but has since been removed. I don't know if this should be tuned.
        ggml_backend_metal_set_n_cb(backend, 1);
#endif
    }
}

ggml_backend_sched_ptr gpu_context::sched(size_t max_nodes) const {
    ggml_backend_t backs[2]{gpu(), cpu.get()};
    const bool cpu_only{backs[0] == backs[1]};
    return ggml_backend_sched_ptr{ggml_backend_sched_new(backs, nullptr, 2 - cpu_only, max_nodes, false)};
}

runner_context::runner_context(const shared_ptr<gpu_context> & gpu, size_t max_nodes)
    : gpu{gpu}, sched{gpu->sched(max_nodes)}, ctx{
          ggml_init({
              .mem_size = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false),
              .mem_buffer = nullptr,
              .no_alloc = true,
          })
      } {
}

tts_model::tts_model(shared_ptr<gpu_context> gpu, const model_tensor_meta & tensor_meta)
    : ctx{
          ggml_init({
              .mem_size = tensor_meta.mem_size,
              .mem_buffer = nullptr,
              .no_alloc = true,
          })
      }, buf{ggml_backend_alloc_buffer(gpu->gpu(), tensor_meta.buf_size)},
      max_nodes{max<size_t>(8192, tensor_meta.n_tensors * 5)}, buf_offset{ggml_tallocr_new(&*buf)} {
    ggml_backend_buffer_set_usage(&*buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
}

ggml_tensor * tts_model::copy_to_gpu(ggml_tensor * src) {
    if (ggml_backend_buffer_is_host(&*buf)) {
        return src;
    }
    ggml_tensor * const result{ggml_dup_tensor(&*ctx, src)};
    ggml_set_name(result, src->name);
    // TODO the scheduler or at least ggml_backend_alloc_ctx_tensors should be doing this for us
    ggml_tallocr_alloc(&buf_offset, result);
    ggml_backend_tensor_copy(src, result);
    return result;
}
