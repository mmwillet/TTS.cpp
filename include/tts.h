#pragma once

#include "tts_model.h"

tts_runner * runner_from_file(str fname, int n_threads, const generation_configuration & config, bool cpu_only = true);
int generate(tts_runner * runner, str sentence, tts_response & response, const generation_configuration & config);
void update_conditional_prompt(tts_runner * runner, str file_path, str prompt, bool cpu_only = true);
