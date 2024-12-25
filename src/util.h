#ifndef util_h
#define util_h

#include <stdio.h>
#include <string>
#include <vector>
#include <stdint.h>
#include <sys/types.h>
#include "ggml-metal.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpp.h"

#define TTS_ABORT(...) tts_abort(__FILE__, __LINE__, __VA_ARGS__)


struct model_tensor_meta {
	uint32_t n_tensors = 0;
	size_t n_bytes = 0;
};

struct model_tensor_meta compute_tensor_meta(std::string name_prefix, ggml_context * weight_ctx);
struct ggml_tensor * dac_snake_1d(ggml_context * ctx, struct ggml_tensor * alpha, struct ggml_tensor * a);
uint64_t get_cpu_count();


bool has_suffix(std::string value, std::string suffix);
bool has_prefix(std::string value, std::string prefix);

void tts_abort(const char * file, int line, const char * fmt, ...);

#endif
