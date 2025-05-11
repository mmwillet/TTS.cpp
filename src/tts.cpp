#include <mutex>
#include "dia_model.h"
#include "kokoro_model.h"
#include "parler_model.h"
#include "tts.h"

struct tts_runner_factory {
    virtual unique_ptr<tts_runner> create(const gguf_context & meta, const ggml_context & weights,
                                          const generation_configuration & config) = 0;

    virtual ~tts_runner_factory() = default;
};

template<typename T>
    requires is_base_of_v<tts_runner, T> && is_base_of_v<tts_model, typename T::model_type>
struct tts_runner_factory_impl final : tts_runner_factory {
    unique_ptr<tts_runner> create(const gguf_context & meta, const ggml_context & weights,
                                  const generation_configuration & config) override {
        const shared_ptr<typename T::model_type> model{meta, weights};
        model->buf_offset = 0;
        // TODO: change this weight assignment pattern to mirror llama.cpp
        model->assign_weights(weights);
        return unique_ptr<tts_runner>{new T{model, config}};
    }
};

static constexpr array<pair<str, unique_ptr<tts_runner_factory>>, 3> MODEL_FACTORIES{{
    {"dia", make_unique<tts_runner_factory_impl<dia_runner>>()},
    {"kokoro", make_unique<tts_runner_factory_impl<kokoro_runner>>()},
    {"parler-tts", make_unique<tts_runner_factory_impl<parler_tts_runner>>()},
}};

// currently only metal and cpu devices are supported, so cpu_only only describes whether or not to try to load and run on metal.
unique_ptr<tts_runner> runner_from_file(str fname, int n_threads, generation_configuration * config, bool cpu_only) {
    ggml_context * weight_ctx_{};
    const gguf_context_ptr meta{gguf_init_from_file(fname, {
        .no_alloc   = false,
        .ctx        = &weight_ctx_,
    })};
    const ggml_context_ptr weights{weight_ctx_};
    if (!meta) {
        TTS_ABORT(__func__ " failed for file %s\n");
    }
    const int arch_key{gguf_find_key(&*meta, "general.architecture")};
    if (arch_key == -1) {
        TTS_ABORT(__func__ " failed for file %s. No architecture is set.\n", fname);
    }
    const sv arch{gguf_get_val_str(&*meta, arch_key)};
    const auto found{binary_search_idx(MODEL_FACTORIES, arch, pair_first_cmp_sv)};
    if (!~found) {
        TTS_ABORT(__func__ " failed for file %s. The architecture '%s' is not supported.", fname, arch);
    }
    return MODEL_FACTORIES[found].second.create(*meta, *weights, config);
}

static bool dac_is_quantizable(sv name, const quantization_params & params) {
    // The DAC audio encoder / decoder is not compatible with quantization
    return !name.starts_with("audio_encoder");
}

static bool dia_is_quantizable(sv name, const quantization_params & params) {
    // Normalization tensors should not be quantized.
    return dac_is_quantizable(name, params) && !name.ends_with("norm") &&
           (params.quantize_output_heads || !name.ends_with("dia.decoder.heads"));
}

static bool parler_is_quanitizable(sv name, const quantization_params & params) {
    // Normalization weights shouldn't be quantized, and the text encoding shouldn't be normalized.
    return dac_is_quantizable(name, params) && !name.ends_with("norm.weight") && !name.ends_with("text_encoding") &&
           !name.ends_with("positional_embed") && !name.ends_with("norm.bias") &&
           (params.quantize_output_heads || !name.ends_with("weight.head")) &&
           (params.quantize_text_embeddings || !name.ends_with("embed_prompts")) &&
           (params.quantize_cross_attn_kv ||
               !name.ends_with("encoder_attn.k_proj.weight") && !name.ends_with("encoder_attn.v_proj.weight"));
}

bool is_quanitizable(char arch, sv name, const quantization_params & params) {
    switch(arch) {
        case 'p':
            return parler_is_quanitizable(name, params);
        case 'd':
            return dia_is_quantizable(name, params);
        default:
            TTS_ABORT(__func__ " failed. The architecture '%d' is not supported.", arch);
    }
}

size_t quantize_tensor(void * new_data, ggml_tensor * tensor, const float * imatrix, enum ggml_type qtype, uint32_t n_threads) {
    // much of this is form copied from llama.cpp
    int chunk_size_multiplier = 1;
    if (qtype == GGML_TYPE_Q4_0_4_4 || qtype == GGML_TYPE_Q4_0_4_8 || qtype == GGML_TYPE_Q4_0_8_8) {
        if ((qtype == GGML_TYPE_Q4_0_8_8) && (tensor->ne[1] % 8 != 0)) qtype = GGML_TYPE_Q4_0;
        else if (tensor->ne[1] % 4 != 0) qtype = GGML_TYPE_Q4_0;
        if (qtype == GGML_TYPE_Q4_0_8_8) chunk_size_multiplier = 8;
        else if (qtype == GGML_TYPE_Q4_0_4_4 || qtype == GGML_TYPE_Q4_0_4_8) chunk_size_multiplier = 4;
    }
    size_t out_size = 0;
    const int32_t d3_step = tensor->ne[0] * tensor->ne[1];
    const int32_t n_per_row = tensor->ne[0];
    const int32_t nrows = tensor->ne[1];
    static const int32_t min_chunk_size = 32 * 512;
    const int32_t chunk_size = (n_per_row >= min_chunk_size ? n_per_row : n_per_row * ((min_chunk_size + n_per_row - 1)/n_per_row)) * chunk_size_multiplier;
    uint32_t thread_count = max(1, min((int)n_threads, (int)(d3_step + chunk_size - 1) / chunk_size));
    mutex mutex;

    for (int32_t d3_index = 0; d3_index < tensor->ne[2]; d3_index++) {
        const float * f32_data_d3 = ((float *) tensor->data) + d3_index * d3_step;
        void * new_data_d3 = (char *)new_data + ggml_row_size(qtype, tensor->ne[0]) * d3_index * nrows;
        const float * imatrix_03 = imatrix ? imatrix + d3_index * tensor->ne[0] : nullptr;
        if (thread_count <= 1) {
            // not threaded
            out_size += ggml_quantize_chunk(qtype, f32_data_d3, new_data_d3, 0, nrows, n_per_row, imatrix);
        } else {
            vector <thread> threads;
            int64_t counter = 0;
            size_t new_size = 0;
            bool valid = true;
            for (uint32_t t = 0; t < thread_count; t++) {
                auto func = [&mutex, &counter, &new_size, &valid, qtype, f32_data_d3, new_data_d3, chunk_size, nrows, n_per_row, imatrix]() {
                    const int64_t nrows_per_chunk = chunk_size / n_per_row;
                    size_t local_size = 0;
                    while (true) {
                        unique_lock<mutex> lock(mutex);
                        int64_t first_row = counter; 
                        counter += nrows_per_chunk;
                        if (first_row >= nrows) {
                            if (local_size > 0) {
                                new_size += local_size;
                            }
                            break;
                        }
                        lock.unlock();
                        const int64_t this_nrow = min(nrows - first_row, nrows_per_chunk);
                        size_t this_size = ggml_quantize_chunk(qtype, f32_data_d3, new_data_d3, first_row * n_per_row, this_nrow, n_per_row, imatrix);
                        local_size += this_size;

                        // validate the quantized data; I am not sure how this would occur, but there is always the safe fallback on doing this single threaded.
                        const size_t row_size  = ggml_row_size(qtype, n_per_row);
                        void * this_data = (char *) new_data_d3 + first_row * row_size;
                        if (!ggml_validate_row_data(qtype, this_data, this_size)) {
                            unique_lock<mutex> lock(mutex);
                            valid = false;
                            break;
                        }
                    }
                };
                threads.push_back(thread(func));
            }
            for (auto & t : threads) t.join();

            if (!valid) {
                TTS_ABORT("Validation of quantized data failed. Please try again and/or switch to single thread quantization.\n");
            }
            out_size += new_size;
        }
    }
    return out_size;
}

static void zeros(ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

template <typename T>
struct no_init {
    T value;
};

void quantize_gguf(str ifile, str ofile, const quantization_params & params) {
    ggml_context * weight_ctx = NULL;
    gguf_init_params gguf_params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &weight_ctx,
    };
    gguf_context * meta_ctx = gguf_init_from_file(ifile, gguf_params);
    str arch{"parler-tts"}; // only parler-tts gguf files should lack an explicit architecture.
    if (const int arch_key{gguf_find_key(meta_ctx, "general.architecture")}; ~arch_key) {
        arch = gguf_get_val_str(meta_ctx, arch_key);
    }
    if (arch == "kokoro"sv) {
        TTS_ABORT("ERROR: quantization for arch 'kokoro' is not currently support\n");
    }
    switch (params.quantize_type) {
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_F16:
            break;
        default:
            fprintf(stdout, "Warning, %s is untested for quantization type '%d'. Use at your own risk.\n", arch, params.quantize_type);
    }

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    gguf_context_ptr ctx_out{gguf_init_empty()};

    // copy the KV pairs from the input file
    gguf_set_kv(ctx_out.get(), meta_ctx);
    gguf_set_val_u32(ctx_out.get(), "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out.get(), "general.quantization_type", params->quantize_type);
    for (ggml_tensor * tensor = ggml_get_first_tensor(weight_ctx); tensor; tensor = ggml_get_next_tensor(weight_ctx, tensor)) {
        string name = ggml_get_name(tensor);
        if (name.size() != 0) {
            gguf_add_tensor(ctx_out.get(), tensor);
        }
    }

    vector<no_init<uint8_t>> work;

    ofstream fout;
    auto close_ofstream = [&]() {
        // Write metadata and close file handler
        if (fout.is_open()) {
            fout.seekp(0);
            vector<uint8_t> data(gguf_get_meta_size(ctx_out.get()));
            gguf_get_meta_data(ctx_out.get(), data.data());
            fout.write((const char *) data.data(), data.size());
            fout.close();
        }
    };
    auto new_ofstream = [&]() {
        string fname = ofile;
        fout = ofstream(fname, ios::binary);
        fout.exceptions(ofstream::failbit); // fail fast on write errors
        const size_t meta_size = gguf_get_meta_size(ctx_out.get());
        // placeholder for the meta data
        ::zeros(fout, meta_size);
    };
    new_ofstream();
    for (ggml_tensor * cur = ggml_get_first_tensor(weight_ctx); cur; cur = ggml_get_next_tensor(weight_ctx, cur)) {
        ggml_type new_type;
        void * new_data;
        size_t new_size;
        string name = ggml_get_name(cur);
        
        if (name.size() == 0) {
            continue;
        }

        if (is_quanitizable(arch[0], name, params)) {
            if ((cur->type) != GGML_TYPE_F32) {
                TTS_ABORT("ERROR: All quantized tensors must be transformed from 32bit floats. Tensor, '%s', has improper type, '%d'\n", cur->name, cur->type);
            }
            new_type = params.quantize_type;
            if ((new_type >= GGML_TYPE_IQ2_XXS && new_type <= GGML_TYPE_IQ4_XS)) {
                TTS_ABORT("ERROR: Quantization type '%d' requires an importance matrix.\n", new_type);
            }
            const int64_t nelement_size = ggml_nelements(cur) * 4;
            if (work.size() < (size_t)nelement_size) {
                work.resize(nelement_size); // upper bound on size
            }
            new_data = work.data();
            new_size = quantize_tensor(new_data, cur, nullptr, new_type, params.n_threads);
        } else if (params.convert_dac_to_f16 && name.starts_with("audio_encoder") && !name.ends_with("alpha") && !name.ends_with("bias")) {
            if ((cur->type) != GGML_TYPE_F32) {
                TTS_ABORT("ERROR: All converted tensors must be transformed from 32bit floats. Tensor, '%s', has improper type, '%d'\n", cur->name, cur->type);
            }
            new_type = GGML_TYPE_F16;
            const int64_t nelement_size = ggml_nelements(cur) * 4;
            if (work.size() < (size_t)nelement_size) {
                work.resize(nelement_size); // upper bound on size
            }
            new_data = work.data();
            new_size = quantize_tensor(new_data, cur, nullptr, new_type, params.n_threads);
        } else {
            new_type = cur->type;
            new_data = cur->data;
            new_size = ggml_nbytes(cur);
        }

        gguf_set_tensor_type(ctx_out.get(), name.c_str(), new_type);
        gguf_set_tensor_data(ctx_out.get(), name.c_str(), new_data, new_size);
        fprintf(stdout, "At tensor: '%s' with new size: %zu bytes\n", name.c_str(), new_size);
        // write tensor data + padding
        fout.write((const char *) new_data, new_size);
        zeros(fout, GGML_PAD(new_size, align) - new_size);
    }
    close_ofstream();
}
