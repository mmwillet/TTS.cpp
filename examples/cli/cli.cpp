#include "tts.h"
#include "audio_file.h"
#include "args.h"
#include "common.h"

void write_audio_file(std::string path, struct tts_response * data, float sample_rate = 44100.f, float frequency = 440.f, int channels = 1) {
    AudioFile<float> file;
    file.setBitDepth(16);
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

int main(int argc, const char ** argv) {
    float default_temperature = 0.9f;
    int default_n_threads = 10;
    int default_top_k = 50;
    float default_repetition_penalty = 1.1f;
    arg_list args;
    args.add_argument(string_arg("--model-path", "(REQUIRED) The local path of the gguf model file for Parler TTS mini or large v1.", "-mp", true));
    args.add_argument(string_arg("--prompt", "(REQUIRED) The text prompt for which to generate audio in quotation markers.", "-p", true));
    args.add_argument(string_arg("--save-path", "(REQUIRED) The path to save the audio output to in a .wav format.", "-sp", true));
    args.add_argument(float_arg("--temperature", "The temperature to use when generating outputs. Defaults to 0.9.", "-t", false, &default_temperature));
    args.add_argument(int_arg("--n-threads", "The number of cpu threads to run generation with. Defaults to 10.", "-nt", false, &default_n_threads));
    args.add_argument(int_arg("--topk", "(OPTIONAL) When set to an integer value greater than 0 generation uses nucleus sampling over topk nucleaus size. Defaults to 50.", "-tk", false, &default_top_k));
    args.add_argument(float_arg("--repetition-penalty", "The by channel repetition penalty to be applied the sampled output of the model. defaults to 1.1.", "-r", false, &default_repetition_penalty));
    args.add_argument(bool_arg("--use-metal", "(OPTIONAL) Whether to use metal acceleration", "-m"));
    args.add_argument(bool_arg("--no-cross-attn", "(OPTIONAL) Whether to not include cross attention", "-ca"));
    args.add_argument(string_arg("--conditional-prompt", "(OPTIONAL) A distinct conditional prompt to use for generating. If none is provided the preencoded prompt is used. '--text-encoder-path' must be set to use conditional generation.", "-cp", false));
    args.add_argument(string_arg("--text-encoder-path", "(OPTIONAL) The local path of the text encoder gguf model for conditional generaiton.", "-tep", false));
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

    generation_configuration * config = new generation_configuration(*args.get_int_param("--topk"), *args.get_float_param("--temperature"), *args.get_float_param("--repetition-penalty"), !args.get_bool_param("--no-cross-attn"));

    struct tts_runner * runner = runner_from_file(args.get_string_param("--model-path"), *args.get_int_param("--n-threads"), config, !args.get_bool_param("--use-metal"));

    if (conditional_prompt.size() > 0) {
        update_conditional_prompt(runner, text_encoder_path, conditional_prompt, true);
    }
    tts_response data;
    
    generate(runner, args.get_string_param("--prompt"), &data, config);
    write_audio_file(args.get_string_param("--save-path"), &data);
    return 0;
}
