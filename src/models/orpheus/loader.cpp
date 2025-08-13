#include "model.h"
#include "tts.h"

tts_runner * orpheus_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads,
                               generation_configuration * config, tts_arch arch, bool cpu_only) {
    orpheus_model * model       = new orpheus_model;
    snac_model *    audio_model = new snac_model;
    bpe_tokenizer * bt          = bpe_tokenizer_from_gguf(meta_ctx);
    model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
    audio_model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
    sampler *          samp          = new sampler;
    snac_context *     sctx          = build_new_snac_context(audio_model, n_threads, cpu_only);
    snac_runner *      audio_decoder = new snac_runner(audio_model, sctx);
    orpheus_context *  octx          = build_new_orpheus_context(model, n_threads, cpu_only);
    orpheus_kv_cache * cache         = new orpheus_kv_cache;
    orpheus_runner *   runner        = new orpheus_runner(model, audio_decoder, octx, bt, samp, cache);

    for (ggml_tensor * cur = ggml_get_first_tensor(weight_ctx); cur; cur = ggml_get_next_tensor(weight_ctx, cur)) {
        runner->assign_weight(cur->name, cur);
    }

    runner->prepare_post_load();

    gguf_free(meta_ctx);
    ggml_free(weight_ctx);
    runner->arch = arch;

    return runner;
}
