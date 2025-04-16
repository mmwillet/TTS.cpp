#include "tts.h"
#include "args.h"
#include "common.h"
#include "playback.h"
#include "write_file.h"
#include <thread>

int main(int argc, const char ** argv) {
    float default_temperature = 1.0f;
    int default_n_threads = std::min((int)std::thread::hardware_concurrency(), 1);
    int default_top_k = 50;
    float default_repetition_penalty = 1.1f;
    float default_eos_threshold = 0.0f;
    int default_max_eos_tokens = 0;
    arg_list args;
    args.add_argument(string_arg("--model-path", "(REQUIRED) The local path of the gguf model file for Parler TTS mini or large v1.", "-mp", true));
    args.add_argument(string_arg("--prompt", "(REQUIRED) The text prompt for which to generate audio in quotation markers.", "-p", true));
    args.add_argument(string_arg("--save-path", "(OPTIONAL) The path to save the audio output to in a .wav format. Defaults to TTS.cpp.wav", "-sp", false, "TTS.cpp.wav"));
    args.add_argument(float_arg("--temperature", "The temperature to use when generating outputs. Defaults to 1.0.", "-t", false, &default_temperature));
    args.add_argument(int_arg("--n-threads", "The number of cpu threads to run generation with. Defaults to hardware hardware_concurrency (or 1 if hardware concurrency isn't determined).", "-nt", false, &default_n_threads));
    args.add_argument(int_arg("--topk", "(OPTIONAL) When set to an integer value greater than 0 generation uses nucleus sampling over topk nucleaus size. Defaults to 50.", "-tk", false, &default_top_k));
    args.add_argument(float_arg("--repetition-penalty", "The by channel repetition penalty to be applied the sampled output of the model. defaults to 1.1.", "-r", false, &default_repetition_penalty));
    args.add_argument(bool_arg("--use-metal", "(OPTIONAL) Whether to use metal acceleration", "-m"));
    args.add_argument(bool_arg("--no-cross-attn", "(OPTIONAL) Whether to not include cross attention", "-ca"));
    args.add_argument(string_arg("--conditional-prompt", "(OPTIONAL) A distinct conditional prompt to use for generating. If none is provided the preencoded prompt is used. '--text-encoder-path' must be set to use conditional generation.", "-cp", false));
    args.add_argument(string_arg("--text-encoder-path", "(OPTIONAL) The local path of the text encoder gguf model for conditional generaiton.", "-tep", false));
    args.add_argument(string_arg("--voice", "(OPTIONAL) The voice to use to generate the audio. This is only used for models with voice packs.", "-v", false, "af_alloy"));
    args.add_argument(float_arg("--eos-threshold", "(OPTIONAL) the threshold at which point to prioritize an eos token when sampling. When not set or set to 0.0 no eos threshold will not be used.", "-bt", false, &default_eos_threshold));
    args.add_argument(int_arg("--max-eos-tokens", "(OPTIONAL) the total number of eos tokens across all output heads necessary to incur immediate generation stopping. By default this setting is not used. It will only be used if set to a value greater than 0.", "-met", false, &default_max_eos_tokens));
    register_play_tts_response_args(args);
    args.parse(argc, argv);
    if (args.for_help) {
        args.help();
        return 0;
    }
    args.validate();

    std::string conditional_prompt = args.get_string_param("--conditional-prompt");
    std::string text_encoder_path = args.get_string_param("--text-encoder-path");
    if (conditional_prompt.size() > 0 && text_encoder_path.size() <= 0) {
        fprintf(stderr, "The '--text-encoder-path' must be specified when '--condtional-prompt' is passed.\n");
        exit(1);
    }

    generation_configuration * config = new generation_configuration(
        args.get_string_param("--voice"), 
        *args.get_int_param("--topk"), 
        *args.get_float_param("--temperature"), 
        *args.get_float_param("--repetition-penalty"), 
        !args.get_bool_param("--no-cross-attn"), 
        *args.get_float_param("--eos-threshold"),
        *args.get_int_param("--max-eos-tokens"));

    struct tts_runner * runner = runner_from_file(args.get_string_param("--model-path"), *args.get_int_param("--n-threads"), config, !args.get_bool_param("--use-metal"));

    if (conditional_prompt.size() > 0) {
        update_conditional_prompt(runner, text_encoder_path, conditional_prompt, true);
    }
    tts_response data;

    generate(runner, args.get_string_param("--prompt"), &data, config);
    if (!play_tts_response(args, data, runner->sampling_rate)) {
        write_audio_file(data, args.get_string_param("--save-path"), runner->sampling_rate);
    }
    return 0;
}
