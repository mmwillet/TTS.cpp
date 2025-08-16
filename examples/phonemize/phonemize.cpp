#include <stdio.h>

#include "../../src/models/kokoro/phonemizer.h"
#include "args.h"

int main(int argc, const char ** argv) {
    arg_list args;
    args.add_argument(string_arg("--phonemizer-path", "(OPTIONAL) The local path of the gguf phonemiser file for TTS.cpp phonemizer. This is required if not using espeak.", "-mp"));
    args.add_argument(string_arg("--prompt", "(REQUIRED) The text prompt to phonemize.", "-p", true));
    args.add_argument(bool_arg("--use-espeak", "(OPTIONAL) Whether to use espeak to generate phonems.", "-ue"));
    args.add_argument(string_arg("--espeak-voice-id", "(OPTIONAL) The voice id to use for espeak phonemization. Defaults to 'gmw/en-US'.", "-eid", false, "gmw/en-US"));
    args.parse(argc, argv);
    if (args.for_help) {
        args.help();
        return 0;
    }
    args.validate();
    if (!args.get_bool_param("--use-espeak") && args.get_string_param("--phonemizer-path") == "") {
        fprintf(stderr, "The '--phonemizer-path' must be specified when '--use-espeak' is not true.\n");
        exit(1);
    }

    phonemizer * ph;
    if (args.get_bool_param("--use-espeak")) {
        ph = espeak_phonemizer(false, args.get_string_param("--espeak-voice-id"));
    } else {
        ph = phonemizer_from_file(args.get_string_param("--phonemizer-path"));
    }
    std::string response = ph->text_to_phonemes(args.get_string_param("--prompt"));
    fprintf(stdout, "%s\n", response.c_str());
    return 0;
}
