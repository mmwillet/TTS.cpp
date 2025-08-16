#include "model.h"

void dummy_register() {}

dummy_model_loader::dummy_model_loader() : tts_model_loader{ "dummy", true } {}

unique_ptr<tts_generation_runner> dummy_model_loader::from_file(gguf_context *, ggml_context *, int, bool,
                                                                const generation_configuration & config) const {
    return make_unique<dummy_runner>();
}

const dummy_model_loader dummy_loader{};
