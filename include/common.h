#ifndef common_h
#define common_h

#include <string>
#include <map>
#include <vector>

// Using this simple struct as opposed to a common std::vector allows us to return the cpu buffer
// pointer directly rather than copying the contents of the buffer to a predefined std::vector.
struct tts_response {
	float * data;
	size_t n_outputs;
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

struct generation_configuration {
    generation_configuration(
    	std::string voice = "",
    	int top_k = 50, 
    	float temperature = 1.0f, 
    	float repetition_penalty = 1.0f, 
    	bool use_cross_attn = true, 
    	float eos_threshold = 0.0f,
    	int max_eos_tokens = 0,
    	bool sample = true): top_k(top_k), temperature(temperature), repetition_penalty(repetition_penalty), use_cross_attn(use_cross_attn), sample(sample), voice(voice), eos_threshold(eos_threshold), max_eos_tokens(max_eos_tokens) {};

    int max_eos_tokens;
    float eos_threshold;
    bool use_cross_attn;
    float temperature;
    float repetition_penalty;
    int top_k;
    std::string voice = "";
    bool sample = true;
};

struct tts_runner {
	tts_arch arch;
	struct ggml_context * ctx = nullptr;
	float sampling_rate = 44100.0f;

	void init_build(std::vector<uint8_t>* buf_compute_meta);
	void free_build();
};

#endif
