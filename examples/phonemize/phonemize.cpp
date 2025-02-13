#include "phonemizer.h"
#include "args.h"
#include <stdio.h>

int main(int argc, const char ** argv) {
    arg_list args;
    args.add_argument(string_arg("--phonemizer-path", "(REQUIRED) The local path of the gguf phonemiser file for TTS.cpp phonemizer.", "-mp", true));
    args.add_argument(string_arg("--prompt", "(REQUIRED) The text prompt to phonemize.", "-p", true));
    args.parse(argc, argv);
    if (args.for_help) {
        args.help();
        return 0;
    }
    args.validate();
    phonemizer * ph = phonemizer_from_file(args.get_string_param("--phonemizer-path"));
    std::string response = ph->text_to_phonemes(args.get_string_param("--prompt"));
    fprintf(stdout, "%s\n", response.c_str());
    return 0;
}
