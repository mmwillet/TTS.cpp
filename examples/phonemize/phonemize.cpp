#include <cstdio>

#include "args_common.h"
#include "phonemizer.h"

int main(int argc, const char ** argv) {
    arg_list args{};
    args.add({"", "prompt", "p", "The text prompt to phonemize", true});
    args.add({
        "", "phonemizer-path", "mp",
        "The local path of the gguf phonemiser file for TTS.cpp phonemizer. "
        "Omit this to use eSpeak to generate phonemes"
    });
    add_espeak_voice_arg(args);
    args.parse(argc, argv);
    const str phonemizer_path{args["phonemizer-path"]};

    phonemizer * ph;
    if (*phonemizer_path) {
        ph = phonemizer_from_file(phonemizer_path);
    } else {
        ph = espeak_phonemizer(false, args["espeak-voice-id"]);
    }
    const string response{ph->text_to_phonemes(string{args["prompt"]})};
    fprintf(stdout, "%s\n", response.c_str());
    return 0;
}
