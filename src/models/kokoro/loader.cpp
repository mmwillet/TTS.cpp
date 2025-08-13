#include "model.h"
#include "tts.h"

tts_runner * kokoro_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads,
                              generation_configuration * config, tts_arch arch, bool cpu_only) {
    kokoro_model *          model = new kokoro_model;
    single_pass_tokenizer * spt   = single_pass_tokenizer_from_gguf(meta_ctx, "tokenizer.ggml.tokens");
    model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
    kokoro_duration_context * kdctx           = build_new_duration_kokoro_context(model, n_threads, cpu_only);
    kokoro_duration_runner *  duration_runner = new kokoro_duration_runner(model, kdctx, spt);
    kokoro_context *          kctx            = build_new_kokoro_context(model, n_threads, cpu_only);
    // if an espeak voice id wasn't specifically set infer it from the kokoro voice, if it was override it, otherwise fallback to American English.
    std::string               espeak_voice_id = config->espeak_voice_id;
    if (espeak_voice_id.empty()) {
        espeak_voice_id = !config->voice.empty() &&
                                  KOKORO_LANG_TO_ESPEAK_ID.find(config->voice.at(0)) != KOKORO_LANG_TO_ESPEAK_ID.end() ?
                              KOKORO_LANG_TO_ESPEAK_ID[config->voice.at(0)] :
                              "gmw/en-US";
    }
    phonemizer *    phmzr  = phonemizer_from_gguf(meta_ctx, espeak_voice_id);
    kokoro_runner * runner = new kokoro_runner(model, kctx, spt, duration_runner, phmzr);

    // TODO: change this weight assignment pattern to mirror llama.cpp
    for (ggml_tensor * cur = ggml_get_first_tensor(weight_ctx); cur; cur = ggml_get_next_tensor(weight_ctx, cur)) {
        runner->assign_weight(cur->name, cur);
    }

    runner->prepare_post_load();

    gguf_free(meta_ctx);
    ggml_free(weight_ctx);
    runner->arch = arch;

    return runner;
}
