#include "../loaders.h"
#include "model.h"

void parler_register() {}

parler_model_loader::parler_model_loader() : tts_model_loader{ "parler-tts" } {}

unique_ptr<tts_generation_runner> parler_model_loader::from_file(gguf_context * meta_ctx, ggml_context * weight_ctx,
                                                                 int n_threads, bool cpu_only,
                                                                 const generation_configuration & config) const {
    parler_tts_model *  model       = new parler_tts_model;
    dac_model *         audio_model = new dac_model;
    unigram_tokenizer * ut          = unigram_tokenizer_from_gguf(meta_ctx);
    ut->initialize_tokenizer();
    model->use_cross_attn = config.use_cross_attn;
    model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
    audio_model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
    sampler *         samp          = new sampler;
    dac_context *     dctx          = build_new_dac_context(audio_model, n_threads, cpu_only);
    dac_runner *      audio_decoder = new dac_runner(audio_model, dctx);
    parler_context *  pctx          = build_new_parler_context(model, n_threads, cpu_only);
    parler_kv_cache * cache         = new parler_kv_cache;
    return make_unique<parler_tts_runner>(model, audio_decoder, pctx, ut, samp, cache);
}

const parler_model_loader parler_loader{};
