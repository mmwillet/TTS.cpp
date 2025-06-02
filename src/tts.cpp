#include "tts.h"
#include <mutex>
#include "dia_model.h"
#include "kokoro_model.h"
#include "parler_model.h"

struct tts_runner * parler_tts_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads, generation_configuration * config, tts_arch arch, bool cpu_only) {
    parler_tts_model * model = new parler_tts_model;
    dac_model * audio_model = new dac_model;
    unigram_tokenizer * ut = unigram_tokenizer_from_gguf(meta_ctx);
    ut->initialize_tokenizer();
    model->use_cross_attn = config->use_cross_attn;
    model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
    audio_model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
    struct sampler * samp = new sampler;
    struct dac_context * dctx = build_new_dac_context(audio_model, n_threads, cpu_only);
    struct dac_runner * audio_decoder = new dac_runner(audio_model, dctx);
    struct parler_context * pctx = build_new_parler_context(model, n_threads, cpu_only);
    struct parler_kv_cache * cache = new parler_kv_cache;
    struct parler_tts_runner * runner = new parler_tts_runner(model, audio_decoder, pctx, ut, samp, cache);

    // TODO: change this weight assignment pattern to mirror llama.cpp
    for (ggml_tensor * cur = ggml_get_first_tensor(weight_ctx); cur; cur = ggml_get_next_tensor(weight_ctx, cur)) {
        runner->assign_weight(cur->name, cur);
    }

    if (config->use_cross_attn) {
        runner->model->prep_cross_key_values(n_threads);
    }

    runner->prepare_post_load();

    gguf_free(meta_ctx);
    ggml_free(weight_ctx);
    runner->arch = arch;

    return runner;
}

struct tts_runner * kokoro_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads, generation_configuration * config, tts_arch arch, bool cpu_only) {
    unique_ptr<kokoro_model> model = make_unique<kokoro_model>();
    single_pass_tokenizer * spt = single_pass_tokenizer_from_gguf(meta_ctx, "tokenizer.ggml.tokens");
    model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
    kokoro_duration_context * kdctx = build_new_duration_kokoro_context(&*model, n_threads, cpu_only);
    kokoro_duration_runner * duration_runner = new kokoro_duration_runner(&*model, kdctx, spt);
    kokoro_context * kctx = build_new_kokoro_context(&*model, n_threads, cpu_only);
    // if an espeak voice id wasn't specifically set infer it from the kokoro voice, if it was override it, otherwise fallback to American English.
    std::string espeak_voice_id = config->espeak_voice_id;
    if (espeak_voice_id.empty()) {
        espeak_voice_id = get_espeak_id_from_kokoro_voice(config->voice.c_str());
    }
    phonemizer * phmzr = phonemizer_from_gguf(meta_ctx, espeak_voice_id);
    kokoro_runner * runner = new kokoro_runner(move(model), kctx, spt, duration_runner, phmzr);

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

struct tts_runner * dia_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads, generation_configuration * config, tts_arch arch, bool cpu_only) {
    dia_model * model = new dia_model;
    dac_model * audio_model = new dac_model;
    model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
    audio_model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
    struct sampler * samp = new sampler;
    struct dac_context * dctx = build_new_dac_context(audio_model, n_threads, cpu_only);
    struct dac_runner * audio_decoder = new dac_runner(audio_model, dctx);
    struct dia_context * diactx = build_new_dia_context(model, n_threads, cpu_only);
    struct dia_kv_cache * cache = new dia_kv_cache;
    struct dia_runner * runner = new dia_runner(model, audio_decoder, diactx, samp, cache);

    for (ggml_tensor * cur = ggml_get_first_tensor(weight_ctx); cur; cur = ggml_get_next_tensor(weight_ctx, cur)) {
        runner->assign_weight(cur->name, cur);
    }

    runner->prepare_post_load();

    gguf_free(meta_ctx);
    ggml_free(weight_ctx);
    runner->arch = arch;

    return runner;
}

// currently only metal and cpu devices are supported, so cpu_only only describes whether or not to try to load and run on metal.
struct tts_runner * runner_from_file(const std::string & fname, int n_threads, generation_configuration * config, bool cpu_only) {
    ggml_context * weight_ctx = NULL;

    struct gguf_init_params params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &weight_ctx,
    };
    gguf_context * meta_ctx = gguf_init_from_file(fname.c_str(), params);
    if (!meta_ctx) {
        TTS_ABORT("%s failed for file %s\n", __func__, fname.c_str());
    }
    int arch_key = gguf_find_key(meta_ctx, "general.architecture");
    if (arch_key == -1) {
        TTS_ABORT("%s failed for file %s. No architecture is set.\n", __func__, fname.c_str());
    }
    const str arch{gguf_get_val_str(meta_ctx, arch_key)};
    switch(const tts_arch arch_type{parse_arch_type(fname.c_str(), arch)}) {
        case PARLER_TTS_ARCH:
            return parler_tts_from_file(meta_ctx, weight_ctx, n_threads, config, arch_type, cpu_only);
        case KOKORO_ARCH:
            return kokoro_from_file(meta_ctx, weight_ctx, n_threads, config, arch_type, cpu_only);
        case DIA_ARCH:
            return dia_from_file(meta_ctx, weight_ctx, n_threads, config, arch_type, cpu_only);
        default:
            TTS_ABORT("%s failed for file %s. The architecture '%s' is not supported.", __func__, fname.c_str(), arch);
    }
}

int generate(tts_runner * runner, std::string sentence, struct tts_response * response, generation_configuration * config) {
    switch(runner->arch) {
        case PARLER_TTS_ARCH:
            ((parler_tts_runner*)runner)->configure_generation(config);
            return ((parler_tts_runner*)runner)->generate(sentence, response);
        case KOKORO_ARCH:
            return ((kokoro_runner*)runner)->generate(sentence, response, config->voice, config->espeak_voice_id);
        case DIA_ARCH:
            ((dia_runner*)runner)->configure_generation(config);
            return ((dia_runner*)runner)->generate(sentence, response);
        default:
            TTS_ABORT("%s failed. The architecture '%d' is not supported.", __func__, runner->arch);
    }
}

void update_conditional_prompt(tts_runner * runner, const std::string file_path, const std::string prompt, bool cpu_only) {
    const auto parler{dynamic_cast<parler_tts_runner *>(runner)};
    if (!parler) {
        fprintf(stderr, "Wrong model for conditional prompt\n");
        return;
    }

    const int n_threads = parler->pctx->n_threads;
    parler->update_conditional_prompt(file_path, prompt, n_threads, cpu_only);
}
