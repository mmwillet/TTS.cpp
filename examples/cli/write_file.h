#pragma once

#include "common.h"

void write_audio_file(std::string path, struct tts_response * data, float sample_rate = 44100.f, int channels = 1);
