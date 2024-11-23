#include "parler.h"
#include "audio_file.h"
#include <stdio.h>


struct arg {
    std::string full_name;
    std::string abbreviation = "";
    std::string description = "";
    bool required = false;
    bool has_param = false;

    std::string help_text() {
        std::string htxt = full_name;
        if (abbreviation != "") {
            htxt += " (" + abbreviation + ")";
        }
        htxt += ": ";
        if (description != "") {
            htxt += description + "\n";
        } else {
            htxt += "is a " + (std::string)(required ? "required " : "optional ") + "parameter.\n";
        }
        return htxt;
    }
};

struct bool_arg : public arg {
    bool_arg(std::string fn, std::string desc = "", std::string abbr = "", bool req = false, bool val = false) {
        full_name = fn;
        description = desc;
        abbreviation = abbr;
        required = req;
        value = val;
    };

    bool value = false;
};

struct string_arg : public arg {
    string_arg(std::string fn, std::string desc = "", std::string abbr = "", bool req = false, std::string val = "") {
        full_name = fn;
        description = desc;
        abbreviation = abbr;
        required = req;
        value = val;
    };
    bool has_param = true;
    std::string value;

    int parse(int argc, const char ** argv) {
        required = false;
        value.assign(argv[0]);
        return 1;
    }
};

struct int_arg : public arg {
    int_arg(std::string fn, std::string desc = "", std::string abbr = "", bool req = false, int * val = nullptr) {
        full_name = fn;
        description = desc;
        abbreviation = abbr;
        required = req;
        value = val;
    };

    int * value;

    int parse(int argc, const char ** argv) {
        if (required) {
            required = false;
        }
        int val = atoi(argv[0]);
        value = &val;
        return 1;
    }

};

struct float_arg : public arg {
    float_arg(std::string fn, std::string desc = "", std::string abbr = "", bool req = false, float * val = nullptr) {
        full_name = fn;
        description = desc;
        abbreviation = abbr;
        required = req;
        value = val;
    };

    bool has_param = true;
    float * value;

    int parse(int argc, const char ** argv) {
        if (required) {
            required = false;
        }
        float val = strtof(argv[0], nullptr);
        value = &val;
        return 1;
    }
};

struct arg_list {
    std::vector<float_arg> fargs;
    std::vector<int_arg> iargs;
    std::vector<bool_arg> bargs;
    std::vector<string_arg> sargs;
    bool for_help = false;

    void add_argument(float_arg arg) {
        fargs.push_back(arg);
    }

    void add_argument(int_arg arg) {
        iargs.push_back(arg);
    }

    void add_argument(bool_arg arg) {
        bargs.push_back(arg);
    }

    void add_argument(string_arg arg) {
        sargs.push_back(arg);
    }

    void help() {
        std::string help_text = "";
        for (auto arg : fargs) {
            help_text += arg.help_text();
        }
        for (auto arg : iargs) {
            help_text += arg.help_text();

        }
        for (auto arg : bargs) {
            help_text += arg.help_text();

        }
        for (auto arg : sargs) {
            help_text += arg.help_text();

        }
        fprintf(stdout, help_text.c_str());
    }

    void validate() {
        for (auto arg : fargs) {
            if (arg.required) {
                fprintf(stderr, "argument '%s' is required.\n", arg.full_name.c_str());
                exit(1);
            }
        }
        for (auto arg : iargs) {
            if (arg.required) {
                fprintf(stderr, "argument '%s' is required.\n", arg.full_name.c_str());
                exit(1);
            }
        }
        for (auto arg : bargs) {
            if (arg.required) {
                fprintf(stderr, "argument '%s' is required.\n", arg.full_name.c_str());
                exit(1);
            }
        }
        for (auto arg : sargs) {
            if (arg.required) {
                fprintf(stderr, "argument '%s' is required.\n", arg.full_name.c_str());
                exit(1);
            }
        }
    }

    void parse(int argc, const char ** argv) {
        int current_arg = 1;
        while (current_arg < argc) {
            std::string name(argv[current_arg]);
            if (name == "--help") {
                for_help = true;
                return;
            }
            current_arg += 1;
            current_arg += find_and_parse(name, argc - current_arg, argv + current_arg);
        }
    }

    int find_and_parse(std::string name, int argc, const char ** argv) {
        for (int i = 0; i < fargs.size(); i++) {
            if (fargs[i].full_name == name || fargs[i].abbreviation == name) {
                return fargs[i].parse(argc, argv);
            }
        }
        for (int i = 0; i < iargs.size(); i++) {
            if (iargs[i].full_name == name || iargs[i].abbreviation == name) {
                return iargs[i].parse(argc, argv);
            }
        }
        for (int i = 0; i < bargs.size(); i++) {
            if (bargs[i].full_name == name || bargs[i].abbreviation == name) {
                bargs[i].value = !bargs[i].value;
                bargs[i].required = false;
                return 0;
            }

        }
        for (int i = 0; i < sargs.size(); i++) {
            if (sargs[i].full_name == name || sargs[i].abbreviation == name) {
                return sargs[i].parse(argc, argv);
            }
        }
        fprintf(stderr, "argument '%s' is not a valid argument. Call '--help' for information on all valid arguments.\n", name.c_str());
        exit(1);
    }

    std::string get_string_param(std::string full_name) {
        for (auto arg : sargs) {
            if (arg.full_name == full_name) {
                return arg.value;
            }
        }
    }

    int * get_int_param(std::string full_name) {
        for (auto arg : iargs) {
            if (arg.full_name == full_name) {
                return arg.value;
            }
        }
    }

    float * get_float_param(std::string full_name) {
        for (auto arg : fargs) {
            if (arg.full_name == full_name) {
                return arg.value;
            }
        }
    }

    bool get_bool_param(std::string full_name) {
        for (auto arg : bargs) {
            if (arg.full_name == full_name) {
                return arg.value;
            }
        }
    }
};

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
    args.add_argument(string_arg("--model-path", "The local path of the gguf model file for Parler TTS mini v1.", "-mp", true));
    args.add_argument(string_arg("--prompt", "The text prompt for which to generate audio.", "-p", true));
    args.add_argument(string_arg("--save-path", "The path to save the audio output to in a .wav format.", "-sp", true));
    args.add_argument(float_arg("--temperature", "The temperature to use when generating outputs.", "-t", false, &default_temperature));
    args.add_argument(int_arg("--n-threads", "The number of cpu threads to run generation with.", "-nt", false, &default_n_threads));
    args.parse(argc, argv);
    if (args.for_help) {
        args.help();
        return 0;
    }

    fprintf(stdout, "%s\n", args.get_string_param("--model-path").c_str());
    args.validate();

    struct parler_tts_runner * runner = runner_from_file(args.get_string_param("--model-path"), *args.get_int_param("--n-threads"), true);
    runner->sampler->temperature = *args.get_float_param("--temperature");
    std::vector<float> data;
    
    runner->generate(args.get_string_param("--prompt"), &data);
    write_audio_file(args.get_string_param("--save-path"), &data);
    return 0;
}
