#pragma once

#include <map>
#include "imports.h"

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
};

constexpr auto SUPPORTED_ARCHITECTURES{[] {
	std::array<str, 3> result{};
	result[PARLER_TTS_ARCH] = "parler-tts";
	result[KOKORO_ARCH] = "kokoro";
	result[DIA_ARCH] = "dia";
	return result;
}()};


constexpr tts_arch parse_arch_type(str fname, str arch) {
	const auto result = ranges::find(SUPPORTED_ARCHITECTURES, sv{arch});
	if (result == SUPPORTED_ARCHITECTURES.end()) {
		TTS_ABORT("%s failed for file %s. The architecture '%s' is not supported.", __func__, fname, arch);
	}
	return static_cast<tts_arch>(distance(SUPPORTED_ARCHITECTURES.cbegin(), result));
};

struct generation_configuration {
    explicit generation_configuration(
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
	tts_arch arch;
	struct ggml_context * ctx = nullptr;
	float sampling_rate = 44100.0f;
	virtual ~tts_runner() = default;

	void init_build(std::vector<uint8_t>* buf_compute_meta);
	void free_build();
};
