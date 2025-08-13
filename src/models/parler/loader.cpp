#include "model.h"
#include "tts.h"

tts_runner * parler_tts_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads,
                                  const generation_configuration & config, tts_arch arch, bool cpu_only) {
    parler_tts_model *  model       = new parler_tts_model;
    dac_model *         audio_model = new dac_model;
    unigram_tokenizer * ut          = unigram_tokenizer_from_gguf(meta_ctx);
    ut->initialize_tokenizer();
    model->use_cross_attn = config.use_cross_attn;
    model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
    audio_model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
    sampler *           samp          = new sampler;
    dac_context *       dctx          = build_new_dac_context(audio_model, n_threads, cpu_only);
    dac_runner *        audio_decoder = new dac_runner(audio_model, dctx);
    parler_context *    pctx          = build_new_parler_context(model, n_threads, cpu_only);
    parler_kv_cache *   cache         = new parler_kv_cache;
    parler_tts_runner * runner        = new parler_tts_runner(model, audio_decoder, pctx, ut, samp, cache);

    // TODO: change this weight assignment pattern to mirror llama.cpp
    for (ggml_tensor * cur = ggml_get_first_tensor(weight_ctx); cur; cur = ggml_get_next_tensor(weight_ctx, cur)) {
        runner->assign_weight(cur->name, cur);
    }

    if (config.use_cross_attn) {
        runner->model->prep_cross_key_values(n_threads);
    }

    runner->prepare_post_load();

    gguf_free(meta_ctx);
    ggml_free(weight_ctx);
    runner->arch = arch;

    return runner;
}
