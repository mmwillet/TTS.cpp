#include "parler.h"
#include "audio_file.h"
#include "args.h"

void write_audio_file(std::string path, std::vector<float> * data, float sample_rate = 44100.f, float frequency = 440.f, int channels = 1) {
    AudioFile<float> file;
    file.setBitDepth(16);
    file.setNumChannels(channels);
    int samples = (int) (data->size() / channels);
    file.setNumSamplesPerChannel(samples);
    for (int channel = 0; channel < channels; channel++) {
        for (int i = 0; i < samples; i++) {
            file.samples[channel][i] = (*data)[i];
        }
    }
    file.save(path, AudioFileFormat::Wave);
}

int main(int argc, const char ** argv) {
    float default_temperature = 0.9f;
    int default_n_threads = 10;
    arg_list args;
    args.add_argument(string_arg("--model-path", "(REQUIRED) The local path of the gguf model file for Parler TTS mini v1.", "-mp", true));
    args.add_argument(string_arg("--prompt", "(REQUIRED) The text prompt for which to generate audio in quotation markers.", "-p", true));
    args.add_argument(string_arg("--save-path", "(REQUIRED) The path to save the audio output to in a .wav format.", "-sp", true));
    args.add_argument(float_arg("--temperature", "The temperature to use when generating outputs. Defaults to 0.9.", "-t", false, &default_temperature));
    args.add_argument(int_arg("--n-threads", "The number of cpu threads to run generation with. Defaults to 10.", "-nt", false, &default_n_threads));
    args.add_argument(bool_arg("--use-metal", "Whether to use metal acceleration", "-m"));
    args.parse(argc, argv);
    if (args.for_help) {
        args.help();
        return 0;
    }
    args.validate();

    struct parler_tts_runner * runner = runner_from_file(args.get_string_param("--model-path"), *args.get_int_param("--n-threads"), !args.get_bool_param("--use-metal"));
    runner->sampler->temperature = *args.get_float_param("--temperature");
    std::vector<float> data;
    
    runner->generate(args.get_string_param("--prompt"), &data);
    write_audio_file(args.get_string_param("--save-path"), &data);
    return 0;
}
