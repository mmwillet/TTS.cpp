#include "loaders.h"

#include <cstring>
#include <unordered_map>

#include "common.h"
#include "ggml-iterator.h"
#include "ggml.h"
#include "llama-mmap.h"

static unordered_map<string_view, reference_wrapper<const tts_model_loader>> LOADERS;

tts_model_loader::tts_model_loader(const char * arch, bool is_test) : arch{ arch }, is_test{ is_test } {
    LOADERS.emplace(arch, ref(*this));
}

void dia_register();
void dummy_register();
void kokoro_register();
void orpheus_register();
void parler_register();

[[maybe_unused]] static bool loaders = [] {
    dia_register();
    dummy_register();
    kokoro_register();
    orpheus_register();
    parler_register();
    return true;
}();

// currently only metal and cpu devices are supported,
// so cpu_only only describes whether or not to try to load and run on metal.
unique_ptr<tts_generation_runner> runner_from_file(const char * fname, int n_threads,
                                                   const generation_configuration & config, bool cpu_only) {
    string_view fname_sv{ fname };
    if (fname_sv.starts_with("test:")) {
        fname_sv.remove_prefix(sizeof("test:") - 1);
        const auto found{LOADERS.find(fname_sv)};
        if (found == LOADERS.end()) {
            GGML_ABORT("Unknown test model/backend %s\n", fname);
        }
        return found->second.get().from_file(nullptr, nullptr, 0, false, config);
    }
    static const bool use_mmap{ !getenv("OLLAMA_NO_MMAP") };  // TODO(danielzgtg) temporary, will be --no-mmap later
    unique_ptr<llama_mmap> in_mmap{};
    if (use_mmap) {
        llama_file in_map_file{ fname, "r" };
        in_mmap = make_unique<llama_mmap>(&in_map_file);
    }
    ggml_context * weight_ctx{};
    gguf_context * meta_ctx = gguf_init_from_file(fname, {
                                                             .no_alloc{ use_mmap },
                                                             .ctx{ &weight_ctx },
                                                         });
    if (!meta_ctx) {
        GGML_ABORT("gguf_init_from_file failed for file %s\n", fname);
    }
    if (use_mmap) {
        const int n{ gguf_get_n_tensors(&*meta_ctx) };
        int       i{};
        void *    in_buffer{ static_cast<char *>(in_mmap->addr()) + gguf_get_data_offset(meta_ctx) };
        for (ggml_tensor & cur : ggml_tensor_iterator{ *weight_ctx }) {
            GGML_ASSERT(i < n);
            GGML_ASSERT(!strcmp(cur.name, gguf_get_tensor_name(&*meta_ctx, i)));
            cur.data = static_cast</*const*/ char *>(in_buffer) + gguf_get_tensor_offset(&*meta_ctx, i);
            ++i;
        }
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
    runner->buf = move(in_mmap);
    return runner;
}
