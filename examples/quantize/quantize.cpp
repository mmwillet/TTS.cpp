#include <vector>
#include <thread>

#include "args_common.h"
#include "ggml.h"
#include "tts.h"
#include "quantize_impl.h"

namespace {
constexpr array VALID_QUANTIZATION_TYPES{
    GGML_TYPE_F16,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q8_0,
};
}

int main(int argc, const char ** argv) {
    arg_list args{};
    add_baseline_args(args);
    args.add({"", "quantized-model-path", "qp", "The path to save the model in a quantized format", true});
    args.add({
        GGML_TYPE_Q4_0, "quantized-type", "qt",
        "The ggml enum of the quantized type to convert compatible model tensors to. For more information see readme. "
        "Defaults to Q4_0 quantization (2)"
    });
    args.add({false, "convert-dac-to-f16", "df", "Whether to convert the DAC audio decoder model to a 16 bit float"});
    args.add({false, "quantize-output-heads", "qh", "Whether to quantize the output heads"});
    args.add({false, "quantize-text-embedding", "qe", "Whether to quantize the input text embededings"});
    args.add({
        false, "quantize-cross-attn-kv", "qkv",
        "Whether to quantize the cross attention keys and values (only applicable for Parler TTS)"
    });
    args.add({
        false, "convert-non-quantized-to-f16", "nqf",
        "Whether or not to convert quantization incompatible tensors to 16 bit precision. "
        "Only currently applicable to Kokoro"
    });
    args.parse(argc, argv);
    const quantization_params qp{
        .n_threads{static_cast<uint32_t>(static_cast<int>(args["n-threads"]))},
        .quantize_type{static_cast<ggml_type>(static_cast<int>(args["--quantized-type"]))},
        .quantize_output_heads{args["quantize-output-heads"]},
        .quantize_text_embeddings{args["quantize-text-embedding"]},
        .quantize_cross_attn_kv{args["quantize-cross-attn-kv"]},
        .convert_dac_to_f16{args["convert-dac-to-f16"]},
        .convert_non_quantizable_to_f16{args["convert-non-quantized-to-f16"]}
    };
    TTS_ASSERT(ranges::contains(VALID_QUANTIZATION_TYPES, qp.quantize_type));
    quantize_gguf(args["model-path"], args["--quantized-model-path"], qp);
    return 0;
}
