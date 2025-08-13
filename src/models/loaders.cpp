#include "loaders.h"

#include "../util.h"
#include "dia/model.h"
#include "ggml-iterator.h"
#include "ggml.h"
#include "kokoro/model.h"
#include "orpheus/model.h"
#include "parler/model.h"

tts_generation_runner * parler_tts_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads,
                                             const generation_configuration & config, tts_arch arch, bool cpu_only);
tts_generation_runner * kokoro_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads,
                                         const generation_configuration & config, tts_arch arch, bool cpu_only);
tts_generation_runner * dia_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads,
                                      const generation_configuration & config, tts_arch arch, bool cpu_only);
tts_generation_runner * orpheus_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads,
                                          const generation_configuration & config, tts_arch arch, bool cpu_only);


// currently only metal and cpu devices are supported,
// so cpu_only only describes whether or not to try to load and run on metal.
unique_ptr<tts_generation_runner> runner_from_file(const char * fname, int n_threads, const generation_configuration & config,
                                        bool cpu_only) {
    ggml_context * weight_ctx{};
    gguf_context * meta_ctx = gguf_init_from_file(fname, {
                                                             .no_alloc{},
                                                             .ctx{ &weight_ctx },
                                                         });
    if (!meta_ctx) {
        TTS_ABORT("gguf_init_from_file failed for file %s\n", fname);
    }
    const int          arch_key = gguf_find_key(meta_ctx, "general.architecture");
    const char * const arch{ gguf_get_val_str(meta_ctx, arch_key) };
    if (SUPPORTED_ARCHITECTURES.find(arch) == SUPPORTED_ARCHITECTURES.end()) {
        TTS_ABORT("%s failed for file %s. The architecture '%s' is not supported.", __func__, fname,
                  arch);
    }
    unique_ptr<tts_generation_runner> runner{};
    const tts_arch arch_type{SUPPORTED_ARCHITECTURES.at(arch)};
    switch (arch_type) {
        case PARLER_TTS_ARCH:
            runner.reset(parler_tts_from_file(meta_ctx, weight_ctx, n_threads, config, arch_type, cpu_only));
            break;
        case KOKORO_ARCH:
            runner.reset(kokoro_from_file(meta_ctx, weight_ctx, n_threads, config, arch_type, cpu_only));
            break;
        case DIA_ARCH:
            runner.reset(dia_from_file(meta_ctx, weight_ctx, n_threads, config, arch_type, cpu_only));
            break;
        case ORPHEUS_ARCH:
            runner.reset(orpheus_from_file(meta_ctx, weight_ctx, n_threads, config, arch_type, cpu_only));
            break;
        default:
            TTS_ABORT("%s failed for file %s. The architecture '%s' is not supported.", __func__, fname,
                      arch);
    }
    // TODO(mmwillet): change this weight assignment pattern to mirror llama.cpp
    for (ggml_tensor & cur : ggml_tensor_iterator{ *weight_ctx }) {
        if (!cur.data) {
            continue;
        }
        if (!*cur.name) {
            // handles the top level meta tensor
            continue;
        }
        runner->assign_weight(cur.name, cur);
    }
    runner->prepare_post_load();
    gguf_free(meta_ctx);
    ggml_free(weight_ctx);
    runner->arch = arch_type;
    return runner;
}
