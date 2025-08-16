#include "model.h"

#include <cmath>
#include <cstring>

void dummy_runner::generate(const char * sentence, tts_response & output, const generation_configuration &) {
    static constexpr size_t SAMPLING_RATE = 44100;
    this->sampling_rate                   = SAMPLING_RATE;
    const size_t N{ strlen(sentence) };
    outputs = make_unique_for_overwrite<float[]>(output.n_outputs = N * SAMPLING_RATE);
    for (size_t i{}; i < N; ++i) {
        const float wavelength{static_cast<float>(SAMPLING_RATE / M_PI / 2) / (200 + sentence[i]) };
        float *     buf = &outputs[i * SAMPLING_RATE];
        for (size_t j{}; j < SAMPLING_RATE; ++j) {
            buf[j] = sin(j * static_cast<float>(M_PI / SAMPLING_RATE)) * sin(j / wavelength);
        }
    }
    output.data = outputs.get();
}
