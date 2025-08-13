#pragma once

#include "common.h"

tts_runner * runner_from_file(const std::string & fname, int n_threads, generation_configuration * config, bool cpu_only = true);
int generate(tts_runner * runner, std::string sentence, struct tts_response * response, generation_configuration * config);
void update_conditional_prompt(tts_runner * runner, const std::string file_path, const std::string prompt, bool cpu_only = true);
std::vector<std::string> list_voices(tts_runner * runner);
