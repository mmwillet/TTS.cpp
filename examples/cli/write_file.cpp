#include <cstdint>
#include "audio_file.h"
#include "write_file.h"

void write_audio_file(const tts_response & data, str path, float sample_rate) {
    fprintf(stdout, "Writing audio file: %s\n", path);
    AudioFile<float> file;
    file.setSampleRate(sample_rate);
    file.samples[0] = std::vector(data.data, data.data + data.n_outputs);
    file.save(path, AudioFileFormat::Wave);
    file.printSummary();
}
