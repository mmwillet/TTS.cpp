#include "../loaders.h"
#include "ggml.h"
#include "model.h"

tts_runner * dia_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads,
                           const generation_configuration & config, tts_arch arch, bool cpu_only) {
    dia_model * model       = new dia_model;
    dac_model * audio_model = new dac_model;
    model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
    audio_model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
    sampler *      samp          = new sampler;
    dac_context *  dctx          = build_new_dac_context(audio_model, n_threads, cpu_only);
    dac_runner *   audio_decoder = new dac_runner(audio_model, dctx);
    dia_context *  diactx        = build_new_dia_context(model, n_threads, cpu_only);
    dia_kv_cache * cache         = new dia_kv_cache;
    dia_runner *   runner        = new dia_runner(model, audio_decoder, diactx, samp, cache);
    return runner;
}
