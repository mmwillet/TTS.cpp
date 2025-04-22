#pragma once

#include <math.h>
#include "common.h"

float energy(float * chunk, int count);

void apply_energy_voice_inactivity_detection(
	tts_response & data, 
	float sample_rate = 44100.0f, 
	int ms_per_frame = 10,
	int frame_threshold = 20,
	float normalized_energy_threshold = 0.01f);
