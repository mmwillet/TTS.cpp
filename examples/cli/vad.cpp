#include "vad.h"

float energy(float * chunk, int count) {
	float en = 0.0f;
	for (int i = 0; i < count; i++) {
		en += powf(chunk[i], 2.0f);
	}
	return en;
}

void apply_energy_voice_inactivity_detection(
	tts_response & data, 
	float sample_rate, 
	int ms_per_frame,
	int frame_threshold,
	float normalized_energy_threshold) {
	int samples_per_frame = (int) (ms_per_frame * sample_rate / 1000.0f);
	int n_frames = (int) (data.n_outputs / samples_per_frame);

	// for min-max normalization
	float max_energy = 0.0f;
	float min_energy = 0.0f;
	float * energies = (float *) malloc(n_frames * sizeof(float));

	// compute the energies and the necessary elements for min-max normalization
	for (int i = 0; i < n_frames; i++) {
		float * chunk = data.data + i * samples_per_frame;
		energies[i] = energy(chunk, samples_per_frame);
		if (i == 0) {
			max_energy = energies[i];
			min_energy = energies[i];
		} else if (energies[i] > max_energy) {
			max_energy = energies[i];
		} else if (energies[i] < min_energy) {
			min_energy = energies[i];
		}
	}

	int concurrent_silent_frames = 0;

	for (int i = 0; i < n_frames; i++) {
		float frame_energy = (energies[i] - min_energy) / (max_energy - min_energy);
		if (frame_energy < normalized_energy_threshold) {
			concurrent_silent_frames++;
		} else {
			concurrent_silent_frames = 0;
		}
	}
	if (concurrent_silent_frames >= frame_threshold) {
		data.n_outputs -= ((concurrent_silent_frames - 1) * samples_per_frame);
	}
	free(energies);
}
