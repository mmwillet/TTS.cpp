#include <thread>
#include "args_common.h"
#include "playback.h"
#include "tts.h"
#include "vad.h"
#include "write_file.h"

class tts_timing_printer {
    const int64_t start_us{[] {
        ggml_time_init();
        return ggml_time_us();
    }()};
public:
    ~tts_timing_printer() {
        const int64_t end_us{ggml_time_us()};
        // Just a simple "total time" for now before adding "load" / "prompt eval" / "eval" from llama_print_timings
        printf("total time = %.2f ms\n", (end_us - start_us) / 1000.0f);
    }
};

int main(int argc, const char ** argv) {
    const tts_timing_printer _{};
    arg_list args{};
    add_common_args(args);
    args.add({"", "prompt", "p", "The text prompt for which to generate audio", true});
    args.add({"TTS.cpp.wav", "save-path", "sp", "The path to save the audio output to in a .wav format"});
    args.add({
        "", "conditional-prompt", "cp",
        "A distinct conditional prompt to use for generating. "
        "If none is provided the preencoded prompt is used. "
        "'--text-encoder-path' must be set to use conditional generation"
    });
    add_text_encoder_arg(args);
    args.add({
        false, "vad", "va",
        "Whether to apply voice inactivity detection (VAD) and strip silence form the end of the output. "
        "This is particularly useful for Parler TTS. By default, no VAD is applied"
    });
    register_play_tts_response_args(args);
    args.parse(argc, argv);

    const str conditional_prompt{args["conditional-prompt"]};
    const str text_encoder_path{args["text-encoder-path"]};
    if (*conditional_prompt && !*text_encoder_path) {
        fprintf(stderr, "The '--text-encoder-path' must be specified when '--condtional-prompt' is passed.\n");
        exit(1);
    }

    const generation_configuration config{parse_generation_config(args)};
    tts_runner * const runner{runner_from_args(args, config)};

    if (*conditional_prompt) {
        update_conditional_prompt(runner, text_encoder_path, conditional_prompt, true);
    }
    tts_response data;

    const str prompt{args["prompt"]};
    generate(runner, prompt, data, config);
    if (data.n_outputs == 0) {
        fprintf(stderr, "Got empty response for prompt, '%s'.\n", prompt);
        exit(1);
    }
    if (args["vad"]) {
        apply_energy_voice_inactivity_detection(data, runner->sampling_rate);
    }
    if (!play_tts_response(args, data, runner->sampling_rate)) {
        write_audio_file(data, args["save-path"], runner->sampling_rate);
    }
    return 0;
}
