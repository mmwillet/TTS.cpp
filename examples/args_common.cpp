#include "args_common.h"

#include "tts.h"

void add_baseline_args(arg_list & args) {
    // runner_from_file
    args.add({"", "model-path", "mp", "The local path of the gguf model(s) to load", true});
    args.add({
        max(static_cast<int>(thread::hardware_concurrency()), 1), "n-threads", "nt",
        "The number of CPU threads to run calculations with. Defaults to known hardware concurrency. "
        "If hardware concurrency cannot be determined then it defaults to 1"
    });
}

static constexpr generation_configuration default_config{};

void add_common_args(arg_list & args) {
    add_baseline_args(args);
    // generation_configuration
    args.add({!default_config.use_cross_attn, "no-cross-attn", "ca", "Whether to not include cross attention"});
    args.add({default_config.temperature, "temperature", "t", "The temperature to use when generating outputs"});
    args.add({
        default_config.repetition_penalty, "repetition-penalty", "r",
        "The per-channel repetition penalty to be applied the sampled output of the model"
    });
    args.add({
        default_config.top_p, "top-p", "mt",
        "The sum of probabilities to sample over. Must be a value between 0.0 and 1.0. Defaults to 1.0"
    });
    args.add({
        default_config.top_k, "topk", "tk",
        "When set to an integer value greater than 0 generation uses nucleus sampling over topk nucleus size. "
        "Defaults to 50"
    });
    args.add({
        default_config.max_tokens, "max-tokens", "mt",
        "The max audio tokens or token batches to generate where each represents approximates 11 ms of audio. "
        "Only applied to Dia generation. If set to zero as is its default then the default max generation size. "
        "Warning values under 15 are not supported"
    });
    args.add({
        default_config.voice, "voice", "v",
        "The voice to use to generate the audio. This is only used for models with voice packs"
    });
    add_espeak_voice_arg(args);
    // runner_from_file
    args.add({false, "use-metal", "m", "Whether to use metal acceleration"});
}

generation_configuration parse_generation_config(const arg_list & args) {
    const generation_configuration config{
        .use_cross_attn{!args["no-cross-attn"]},
        .temperature{args["temperature"]},
        .repetition_penalty{args["repetition-penalty"]},
        .top_p{args["top-p"]},
        .top_k{args["topk"]},
        .max_tokens{args["max-tokens"]},
        .voice{args["voice"]},
        .espeak_voice_id{args["espeak-voice-id"]}
    };
    if (config.top_p > 1.0f || config.top_p <= 0.0f) {
        fprintf(stderr, "The '--top-p' value must be between 0.0 and 1.0. It was set to '%.6f'.\n", config.top_p);
        exit(1);
    }
    return config;
}

tts_runner * runner_from_args(const arg_list & args, const generation_configuration & config) {
    return runner_from_file(args["model-path"], args["n-threads"], config, !args["use-metal"]);
}

void add_text_encoder_arg(arg_list & args) {
    args.add({
        "", "text-encoder-path", "tep",
        "The local path of the text encoder gguf model for conditional generation"
    });
}

void add_espeak_voice_arg(arg_list & args) {
    args.add({
        default_config.espeak_voice_id, "espeak-voice-id", "eid",
        "The eSpeak voice id to use for phonemization. "
        "This should only be specified when the correct eSpeak voice cannot be inferred from the Kokoro voice. "
        "See MultiLanguage Configuration in the README for more info"
    });
}
