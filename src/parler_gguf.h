#ifndef parler_gguf_h
#define parler_gguf_h

#include "parler_model.h"
#include "tokenizer.h"

static std::pair<int, std::string> parse_layer_count(std::string name, int skip = 0);

// for DAC loading
void assign_residual_unit(dac_model & model, dac_residual_unit & layer, std::string name, ggml_tensor * tensor);
void assign_dac_layer(dac_model & model, dac_layer & layer, std::string name, ggml_tensor * tensor);
void assign_quantizer_layer(dac_model & model, dac_quantize_layer & layer, std::string name, ggml_tensor * tensor);
void assign_to_audio_encoder(dac_model & model, std::string name, ggml_tensor * tensor);

// for Parler loading
void assign_parler_layer(parler_tts_model * model, parler_layer & layer, std::string name, ggml_tensor * tensor);
void assign_to_decoder(parler_tts_model * model, const std::string name, ggml_tensor * tensor);
void assign_weight(parler_tts_model * model, dac_model & audio_model, std::string name, ggml_tensor * tensor);

// for T5 encoder loading
void assign_to_t5_encoder(t5_encoder * model, const std::string name, ggml_tensor * tensor);
void assign_to_t5_layer(t5_encoder * model, t5_layer & layer, std::string name, ggml_tensor * tensor);

// for tokenizer loading
unigram_tokenizer * tokenizer_from_gguf(gguf_context * meta);

enum dac_tensor {
    DAC_ENCODER_IN_KERNEL,
    DAC_ENCODER_IN_BIAS,
    DAC_ENCODER_OUT_KERNEL,
    DAC_ENCODER_OUT_BIAS,
    DAC_ENCODER_SNAKE_ALPHA,
    DAC_ENCODER_LAYER_SNAKE_ALPHA,
    DAC_ENCODER_LAYER_OUT_KERNEL,
    DAC_ENCODER_LAYER_OUT_BIAS,
    DAC_ENCODER_LAYER_RES_BLK_IN_SNAKE,
    DAC_ENCODER_LAYER_RES_BLK_OUT_SNAKE,
    DAC_ENCODER_LAYER_RES_BLK_IN_KERNEL,
    DAC_ENCODER_LAYER_RES_BLK_OUT_KERNEL,
    DAC_ENCODER_LAYER_RES_BLK_IN_BIAS,
    DAC_ENCODER_LAYER_RES_BLK_OUT_BIAS,
    DAC_QUANTIZER_LAYER_IN_KERNEL,
    DAC_QUANTIZER_LAYER_IN_BIAS,
    DAC_QUANTIZER_LAYER_OUT_KERNEL,
    DAC_QUANTIZER_LAYER_OUT_BIAS,
    DAC_QUANTIZER_LAYER_CODEBOOK
};

enum parler_tensor {
    PARLER_EMBD,
    PARLER_EMBD_PROMPTS,
    PARLER_TEXT_ENCODING,
    PARLER_POSITIONAL_EMBD,
    PARLER_HEAD,
    PARLER_NORM,
    PARLER_NORM_BIAS,
    PARLER_LAYER_SELF_ATTN_Q,
    PARLER_LAYER_SELF_ATTN_K,
    PARLER_LAYER_SELF_ATTN_V,
    PARLER_LAYER_SELF_ATTN_O,
    PARLER_LAYER_SELF_ATTN_NORM,
    PARLER_LAYER_SELF_ATTN_NORM_BIAS,
    PARLER_LAYER_ATTN_Q,
    PARLER_LAYER_ATTN_K,
    PARLER_LAYER_ATTN_V,
    PARLER_LAYER_ATTN_O,
    PARLER_LAYER_ATTN_NORM,
    PARLER_LAYER_ATTN_NORM_BIAS,
    PARLER_LAYER_FC1,
    PARLER_LAYER_FC2,
    PARLER_LAYER_OUT_NORM,
    PARLER_LAYER_OUT_NORM_BIAS,
};

enum t5_tensor {
    T5_EMBD,
    T5_NORM,
    T5_DOWN_PROJ,
    T5_DOWN_PROJ_BIAS,
    T5_RELATIVE_BIAS,
    T5_LAYER_ATTN_Q,
    T5_LAYER_ATTN_K,
    T5_LAYER_ATTN_V,
    T5_LAYER_ATTN_O,
    T5_LAYER_ATTN_NORM,
    T5_LAYER_WI_0,
    T5_LAYER_WI_1,
    T5_LAYER_WO,
    T5_LAYER_OUT_NORM,
};

#endif
