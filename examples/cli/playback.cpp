#include <iostream>
#include <pulse/simple.h>
#include "playback.h"

#ifndef PULSE_INSTALL
void register_play_tts_response_args(arg_list & args) {
    // Hide --play
}

bool play_tts_response(arg_list & args, const tts_response & data, float sample_rate) {
    return false;
}
#else
void register_play_tts_response_args(arg_list & args) {
    args.add_argument(bool_arg("--play", "(OPTIONAL) Whether to play back the audio immediately instead of saving it to file."));
}

bool play_tts_response(arg_list & args, const tts_response & data, float sample_rate) {
    if (!args.get_bool_param("--play")) {
        return false;
    }

    const pa_sample_spec ss{
        .format = PA_SAMPLE_FLOAT32NE,
        .rate = static_cast<unsigned>(sample_rate),
        .channels = 1,
    };
    pa_simple *s{pa_simple_new(
        nullptr, "STT.cpp", PA_STREAM_PLAYBACK, nullptr,
        "STT.cpp Text to speech", &ss, nullptr, nullptr, nullptr
    )};
    if (!s) {
        std::cerr << "pa_simple_new failed" << std::endl;
        exit(1);
    }
    std::cout << "Playing audio: " << data.n_outputs << std::endl;
    if (pa_simple_write(s, data.data, data.n_outputs * sizeof(data.data[0]), nullptr)) {
        std::cerr << "pa_simple_write failed" << std::endl;
        exit(1);
    }
    pa_simple_drain(s, nullptr);
    pa_simple_free(s);
    return true;
}
#endif
