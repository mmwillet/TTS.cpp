#ifndef common_h
#define common_h

#include <cstdint>
#include <string>
#include <map>
#include <vector>

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
};

const std::map<std::string, tts_arch> SUPPORTED_ARCHITECTURES = {
	{ "parler-tts", PARLER_TTS_ARCH },
	{ "kokoro", KOKORO_ARCH },
};

const std::map<tts_arch, std::string> ARCHITECTURE_NAMES = {
	{ PARLER_TTS_ARCH, "parler-tts" },
	{ KOKORO_ARCH, "kokoro" },
};

struct generation_configuration {
    generation_configuration(
    	std::string voice = "",
    	int top_k = 50, 
    	float temperature = 1.0, 
    	float repetition_penalty = 1.0, 
    	bool use_cross_attn = true, 
    	std::string espeak_voice_id = "",
    	bool sample = true): top_k(top_k), temperature(temperature), repetition_penalty(repetition_penalty), use_cross_attn(use_cross_attn), sample(sample), voice(voice), espeak_voice_id(espeak_voice_id) {};

    bool use_cross_attn;
    float temperature;
    float repetition_penalty;
    int top_k;
    std::string voice = "";
    bool sample = true;
    std::string espeak_voice_id = "";
};

struct tts_runner {
	tts_arch arch;
	struct ggml_context * ctx = nullptr;
	float sampling_rate = 44100.0f;
	bool supports_voices = false;

	std::string arch_name() {
		return ARCHITECTURE_NAMES.at(arch);
	}

	void init_build(std::vector<uint8_t>* buf_compute_meta);
	void free_build();
};

#endif
