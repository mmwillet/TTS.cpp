#ifndef util_h
#define util_h

#include <stdio.h>
#include <string>
#include <vector>
#include <stdint.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include "ggml-metal.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "ggml-impl.h"

#define TTS_ABORT(...) tts_abort(__FILE__, __LINE__, __VA_ARGS__)

struct ggml_tensor * reciprocal(ggml_context * ctx, ggml_tensor * a);
struct ggml_tensor * dac_snake_1d(ggml_context * ctx, struct ggml_tensor * alpha, struct ggml_tensor * a);
uint64_t get_cpu_count();

void tts_abort(const char * file, int line, const char * fmt, ...);

#endif
