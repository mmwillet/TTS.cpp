#pragma once

#include <cstdint>
#include <string>
#include <map>
#include <memory>
#include <vector>

using namespace std;

// Using this simple struct as opposed to a common std::vector allows us to return the cpu buffer
// pointer directly rather than copying the contents of the buffer to a predefined std::vector.
struct tts_response {
	float * data;
	size_t n_outputs = 0;
	uint32_t hidden_size; // this parameter is only currently used by the t5_encoder for which n_outputs corresponds to sequence length;
};

enum tts_arch {
	PARLER_TTS_ARCH = 0,
	KOKORO_ARCH = 1,
	DIA_ARCH = 2,
	ORPHEUS_ARCH = 3,
};

const std::map<std::string, tts_arch> SUPPORTED_ARCHITECTURES = {
	{ "parler-tts", PARLER_TTS_ARCH },
	{ "kokoro", KOKORO_ARCH },
	{ "dia", DIA_ARCH },
	{ "orpheus", ORPHEUS_ARCH }
};

/// Given a map from keys to values, creates a new map from values to keys
template<typename K, typename V>
static std::map<V, K> reverse_map(const std::map<K, V>& m) {
    std::map<V, K> r;
    for (const auto& kv : m) {
        r[kv.second] = kv.first;
    }
    return r;
}

const std::map<tts_arch, std::string> ARCHITECTURE_NAMES = reverse_map(SUPPORTED_ARCHITECTURES);

struct generation_configuration {
    generation_configuration(
    	std::string voice = "",
    	int top_k = 50,
    	float temperature = 1.0,
    	float repetition_penalty = 1.0,
    	bool use_cross_attn = true,
    	std::string espeak_voice_id = "",
    	int max_tokens = 0,
    	float top_p = 1.0,
    	bool sample = true): top_k(top_k), temperature(temperature), repetition_penalty(repetition_penalty), use_cross_attn(use_cross_attn), sample(sample), voice(voice), espeak_voice_id(espeak_voice_id), max_tokens(max_tokens), top_p(top_p) {};

    bool use_cross_attn;
    float temperature;
    float repetition_penalty;
    float top_p;
    int top_k;
    int max_tokens;
    std::string voice = "";
    bool sample = true;
    std::string espeak_voice_id = "";
};

struct tts_runner {
	struct ggml_context * ctx = nullptr;
	float sampling_rate = 44100.0f;
	bool supports_voices = false;

    virtual ~tts_runner() = default;

	void init_build(std::vector<uint8_t>* buf_compute_meta);
	void free_build();
};

struct ggml_tensor;
struct tts_model_loader;
struct llama_mmap;

struct tts_generation_runner : tts_runner {
    const reference_wrapper<const tts_model_loader> loader;
    unique_ptr<llama_mmap> buf;
    explicit tts_generation_runner(const tts_model_loader & loader);
    ~tts_generation_runner() override;

    virtual void                assign_weight(const char * name, ggml_tensor & tensor) = 0;
    virtual void                prepare_post_load()                                    = 0;
    virtual vector<string_view> list_voices();
    virtual void                update_conditional_prompt(const char * file_path, const char * prompt);
    virtual void generate(const char * sentence, tts_response & output, const generation_configuration & config) = 0;
};

struct test_tts_generation_runner : tts_generation_runner {
    explicit test_tts_generation_runner(const tts_model_loader & loader);

    void assign_weight(const char * name, ggml_tensor & tensor) final;
    void prepare_post_load() final;
};
