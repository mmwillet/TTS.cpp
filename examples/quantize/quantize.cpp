#include "tts.h"
#include "args.h"
#include <stdio.h>
#include <thread>
#include "ggml.h"
#include <vector>

std::vector<ggml_type> valid_quantization_types = {
    GGML_TYPE_F16,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q8_0,
};

int main(int argc, const char ** argv) {
	int default_quantization = (int) GGML_TYPE_Q4_0;
    int default_n_threads = std::max((int)std::thread::hardware_concurrency(), 1);
    arg_list args;
    args.add_argument(string_arg("--model-path", "(REQUIRED) The local path of the gguf model file for Parler TTS mini v1 to quantize.", "-mp", true));
    args.add_argument(string_arg("--quantized-model-path", "(REQUIRED) The path to save the model in a quantized format.", "-qp", true));
    args.add_argument(int_arg("--quantized-type", "(OPTIONAL) The ggml enum of the quantized type to convert compatible model tensors to. For more information see readme. Defaults to Q4_0 quantization (2).", "-qt", false, &default_quantization));
    args.add_argument(int_arg("--n-threads", "(OPTIONAL) The number of cpu threads to run the quantization process with. Defaults to known hardware concurrency.", "-nt", false, &default_n_threads));
    args.add_argument(bool_arg("--convert-dac-to-f16", "(OPTIONAL) Whether to convert the DAC audio decoder model to a 16 bit float.", "-df"));
    args.add_argument(bool_arg("--quantize-output-heads", "(OPTIONAL) Whether to quantize the output heads. Defaults to false and is true when passed (does not accept a parameter).", "-qh"));
    args.add_argument(bool_arg("--quantize-text-embedding", "(OPTIONAL) Whether to quantize the input text embededings (only applicable for Parler TTS). Defaults to false and is true when passed (does not accept a parameter).", "-qe"));
    args.add_argument(bool_arg("--quantize-cross-attn-kv", "(OPTIONAL) Whether to quantize the cross attention keys and values (only applicable for Parler TTS). Defaults to false and is true when passed (does not accept a parameter).", "-qkv"));
    args.add_argument(bool_arg("--convert-non-quantized-to-f16", "(OPTIONAL) Whether or not to convert quantization incompatible tensors to 16 bit precision. Only currently applicable to Kokoro. defaults to false.", "-nqf"));
    args.parse(argc, argv);
    if (args.for_help) {
        args.help();
        return 0;
    }
    args.validate();
    enum ggml_type qtype = static_cast<ggml_type>(*args.get_int_param("--quantized-type"));
    if (std::find(valid_quantization_types.begin(), valid_quantization_types.end(), qtype) == valid_quantization_types.end()) {
    	fprintf(stderr, "ERROR: %d is not a valid quantization type.\n", qtype);
        exit(1);
    }
    struct quantization_params * qp = new quantization_params((uint32_t) *args.get_int_param("--n-threads"), qtype);
    qp->quantize_output_heads = args.get_bool_param("--quantize-output-heads");
    qp->quantize_text_embeddings = args.get_bool_param("--quantize-text-embedding");
    qp->quantize_cross_attn_kv = args.get_bool_param("--quantize-cross-attn-kv");
    qp->convert_dac_to_f16 = args.get_bool_param("--convert-dac-to-f16");
    qp->convert_non_quantizable_to_f16 = args.get_bool_param("--convert-non-quantized-to-f16");
  	quantize_gguf(args.get_string_param("--model-path"), args.get_string_param("--quantized-model-path"), qp);
    return 0;
}
