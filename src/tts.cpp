#include "tts.h"

#include "ggml.h"
#include "models/dia/model.h"
#include "models/kokoro/model.h"
#include "models/orpheus/model.h"
#include "models/parler/model.h"
#include "util.h"

tts_generation_runner * parler_tts_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads,
                                             const generation_configuration & config, tts_arch arch, bool cpu_only);
tts_generation_runner * kokoro_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads,
                                         const generation_configuration & config, tts_arch arch, bool cpu_only);
tts_generation_runner * dia_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads,
                                      const generation_configuration & config, tts_arch arch, bool cpu_only);
tts_generation_runner * orpheus_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads,
                                          const generation_configuration & config, tts_arch arch, bool cpu_only);

// currently only metal and cpu devices are supported, so cpu_only only describes whether or not to try to load and run on metal.
tts_generation_runner * runner_from_file(const std::string & fname, int n_threads,
                                         const generation_configuration & config, bool cpu_only) {
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
