#pragma once

#include "common.h"

void write_audio_file(const tts_response & data, str path = "TTS.cpp.wav", float sample_rate = 44100.0f);
