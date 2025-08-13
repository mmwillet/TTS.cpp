#include "loaders.h"

#include <cstring>
#include <unordered_map>

#include "common.h"
#include "ggml-iterator.h"
#include "ggml.h"

static unordered_map<string_view, reference_wrapper<const tts_model_loader>> LOADERS;

tts_model_loader::tts_model_loader(const char * arch) : arch{ arch } {
    LOADERS.emplace(arch, ref(*this));
}

void dia_register();
void kokoro_register();
void orpheus_register();
void parler_register();

[[maybe_unused]] static bool loaders = [] {
    dia_register();
    kokoro_register();
    orpheus_register();
    parler_register();
    return true;
}();

// currently only metal and cpu devices are supported,
// so cpu_only only describes whether or not to try to load and run on metal.
unique_ptr<tts_generation_runner> runner_from_file(const char * fname, int n_threads,
                                                   const generation_configuration & config, bool cpu_only) {
    ggml_context * weight_ctx{};
    gguf_context * meta_ctx = gguf_init_from_file(fname, {
                                                             .no_alloc{},
                                                             .ctx{ &weight_ctx },
                                                         });
    if (!meta_ctx) {
        GGML_ABORT("gguf_init_from_file failed for file %s\n", fname);
    }
    const int          arch_key = gguf_find_key(meta_ctx, "general.architecture");
    const char * const arch{ gguf_get_val_str(meta_ctx, arch_key) };
    const auto         found = LOADERS.find(arch);
    if (found == LOADERS.end()) {
        GGML_ABORT("Unknown architecture %s\n", arch);
    }
    const auto &                      loader{ found->second.get() };
    unique_ptr<tts_generation_runner> runner{ loader.from_file(meta_ctx, weight_ctx, n_threads, cpu_only, config) };
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
    GGML_ASSERT(&runner->loader.get() == &loader);
    return runner;
}
