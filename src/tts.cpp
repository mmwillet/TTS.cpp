#include "tts.h"

#include "ggml.h"
#include "models/dia/model.h"
#include "models/kokoro/model.h"
#include "models/orpheus/model.h"
#include "models/parler/model.h"
#include "util.h"

tts_runner * parler_tts_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads,
                                  generation_configuration * config, tts_arch arch, bool cpu_only);
tts_runner * kokoro_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads,
                              generation_configuration * config, tts_arch arch, bool cpu_only);
tts_runner * dia_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads,
                           generation_configuration * config, tts_arch arch, bool cpu_only);
tts_runner * orpheus_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads,
                               generation_configuration * config, tts_arch arch, bool cpu_only);

// currently only metal and cpu devices are supported, so cpu_only only describes whether or not to try to load and run on metal.
tts_runner * runner_from_file(const std::string & fname, int n_threads, generation_configuration * config,
                              bool cpu_only) {
    ggml_context * weight_ctx{};

    gguf_init_params params = {
        .no_alloc{},
        .ctx{ &weight_ctx },
    };
    gguf_context * meta_ctx = gguf_init_from_file(fname.c_str(), params);
    if (!meta_ctx) {
        TTS_ABORT("%s failed for file %s\n", __func__, fname.c_str());
    }
    int arch_key = gguf_find_key(meta_ctx, "general.architecture");
    if (arch_key == -1) {
        TTS_ABORT("%s failed for file %s. No architecture is set.\n", __func__, fname.c_str());
    }
    std::string arch = std::string(gguf_get_val_str(meta_ctx, arch_key));
    if (SUPPORTED_ARCHITECTURES.find(arch) == SUPPORTED_ARCHITECTURES.end()) {
        TTS_ABORT("%s failed for file %s. The architecture '%s' is not supported.", __func__, fname.c_str(),
                  arch.c_str());
    }
    switch (tts_arch arch_type = SUPPORTED_ARCHITECTURES.at(arch)) {
        case PARLER_TTS_ARCH:
            return parler_tts_from_file(meta_ctx, weight_ctx, n_threads, config, arch_type, cpu_only);
        case KOKORO_ARCH:
            return kokoro_from_file(meta_ctx, weight_ctx, n_threads, config, arch_type, cpu_only);
        case DIA_ARCH:
            return dia_from_file(meta_ctx, weight_ctx, n_threads, config, arch_type, cpu_only);
        case ORPHEUS_ARCH:
            return orpheus_from_file(meta_ctx, weight_ctx, n_threads, config, arch_type, cpu_only);
        default:
            TTS_ABORT("%s failed for file %s. The architecture '%s' is not supported.", __func__, fname.c_str(),
                      arch.c_str());
    }
}

int generate(tts_runner * runner, std::string sentence, struct tts_response * response,
             generation_configuration * config) {
    switch (runner->arch) {
        case PARLER_TTS_ARCH:
            ((parler_tts_runner *) runner)->configure_generation(config);
            return ((parler_tts_runner *) runner)->generate(sentence, response);
        case KOKORO_ARCH:
            return ((kokoro_runner *) runner)->generate(sentence, response, config->voice, config->espeak_voice_id);
        case DIA_ARCH:
            ((dia_runner *) runner)->configure_generation(config);
            return ((dia_runner *) runner)->generate(sentence, response);
        case ORPHEUS_ARCH:
            ((orpheus_runner *) runner)->configure_generation(config);
            return ((orpheus_runner *) runner)->generate(sentence, response);
        default:
            TTS_ABORT("%s failed. The architecture '%d' is not supported.", __func__, runner->arch);
    }
}

std::vector<std::string> list_voices(tts_runner * runner) {
    switch (runner->arch) {
        case KOKORO_ARCH:
            return ((kokoro_runner *) runner)->list_voices();
        default:
            TTS_ABORT("%s failed. The architecture '%d' does not support #list_voices supported.", __func__,
                      runner->arch);
    }
}

void update_conditional_prompt(tts_runner * runner, const std::string file_path, const std::string prompt,
                               bool cpu_only) {
    int n_threads = ((parler_tts_runner *) runner)->pctx->n_threads;
    ((parler_tts_runner *) runner)->update_conditional_prompt(file_path, prompt, n_threads, cpu_only);
}
