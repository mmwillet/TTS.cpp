#include <cstdint>
#include "write_file.h"
#include "audio_file.h"

void write_audio_file(std::string path, struct tts_response * data, float sample_rate, int channels) {
    AudioFile<float> file;
    file.setBitDepth(16);
    file.setSampleRate(sample_rate);
    file.setNumChannels(channels);
    int samples = (int) (data->n_outputs / channels);
    file.setNumSamplesPerChannel(samples);
    for (int channel = 0; channel < channels; channel++) {
        for (int i = 0; i < samples; i++) {
            file.samples[channel][i] = data->data[i];
        }
    }
    file.save(path, AudioFileFormat::Wave);
}
