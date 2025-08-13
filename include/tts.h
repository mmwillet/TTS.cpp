#pragma once

#include "common.h"

tts_generation_runner * runner_from_file(const std::string & fname, int n_threads,
                                         const generation_configuration & config, bool cpu_only = true);
