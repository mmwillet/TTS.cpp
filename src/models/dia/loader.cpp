#include "../loaders.h"
#include "model.h"

void dia_register() {}

dia_model_loader::dia_model_loader() : tts_model_loader{ "dia" } {}

unique_ptr<tts_generation_runner> dia_model_loader::from_file(gguf_context * meta_ctx, ggml_context * weight_ctx,
                                                              int n_threads, bool cpu_only,
                                                              const generation_configuration & config) const {
    dia_model * model       = new dia_model;
    dac_model * audio_model = new dac_model;
    model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
    audio_model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
    sampler *      samp          = new sampler;
    dac_context *  dctx          = build_new_dac_context(audio_model, n_threads, cpu_only);
    dac_runner *   audio_decoder = new dac_runner(audio_model, dctx);
    dia_context *  diactx        = build_new_dia_context(model, n_threads, cpu_only);
    dia_kv_cache * cache         = new dia_kv_cache;
    return make_unique<dia_runner>(model, audio_decoder, diactx, samp, cache);
}

const dia_model_loader dia_loader{};
