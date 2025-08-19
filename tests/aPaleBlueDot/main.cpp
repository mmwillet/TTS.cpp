#include <fstream>
#include <iostream>

#include "../../src/models/loaders.h"

int main() {
    generation_configuration config("",     // voice (empty)
                                    30,     // top_k (reduced from 50)
                                    1.0f,   // temperature
                                    1.1f,   // repetition_penalty
                                    false,  // use_cross_attn (disabled to save memory)
                                    "",     // espeak_voice_id (empty)
                                    256,    // max_tokens (reduced from 512)
                                    0.95f,  // top_p
                                    true    // sample
    );

    int  n_threads = 4;     // 4 threads on iOS
    bool cpu_only  = true;  // Force CPU-only mode

    unique_ptr<tts_generation_runner> runner{ runner_from_file(
        (string{ getenv("HOME") } + "/parler-tts-mini-v1-Q5_0.gguf").c_str(),  // And it was the 5-bit quantized model.
        n_threads, config, cpu_only) };
    cout << ifstream{"/proc/self/status"}.rdbuf();
    tts_response response{};
    runner->generate("Hello", response, config);
    cout << ifstream{ "/proc/self/status" }.rdbuf();
}
