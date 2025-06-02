#pragma once

#include "tts_model.h"

tts_runner * runner_from_file(const std::string & fname, int n_threads, generation_configuration * config, bool cpu_only = true);
int generate(tts_runner * runner, std::string sentence, tts_response * response, generation_configuration * config);
void update_conditional_prompt(tts_runner * runner, std::string file_path, std::string prompt, bool cpu_only = true);
