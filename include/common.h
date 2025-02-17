#ifndef common_h
#define common_h

// Using this simple struct as opposed to a common std::vector allows us to return the cpu buffer
// pointer directly rather than copying the contents of the buffer to a predefined std::vector.
struct tts_response {
	float * data;
	size_t n_outputs;
	uint32_t hidden_size; // this parameter is only currently used by the t5_encoder for which n_outputs corresponds to sequence length;
};

struct tts_runner {
	std::string arch;
	struct ggml_context * ctx = nullptr;

	void init_build(std::vector<uint8_t>* buf_compute_meta);
	void free_build();
};

struct generation_configuration {
    generation_configuration(int top_k = 50, float temperature = 1.0, float repetition_penalty = 1.0, bool use_cross_attn = true, bool sample = true): top_k(top_k), temperature(temperature), repetition_penalty(repetition_penalty), use_cross_attn(use_cross_attn), sample(sample) {};
    bool use_cross_attn;
    float temperature;
    float repetition_penalty;
    int top_k;
    bool sample = true;
};

#endif
