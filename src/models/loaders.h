#pragma once

#include "../../include/common.h"

unique_ptr<tts_generation_runner> runner_from_file(const char * fname, int n_threads,
                                         const generation_configuration & config, bool cpu_only = true);
