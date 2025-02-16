#include "dac_model.h"

void dac_model::prep_constants(gguf_context * meta) {
    int output_heads_key = search_for_gguf_keys(meta, {"parler-tts.decoder.output_heads", "output_heads"});
    if (output_heads_key != -1) {
        n_heads = gguf_get_val_u32(meta, output_heads_key);;
    }

    int sampling_factor_key = search_for_gguf_keys(meta, {"dac.up_sampling_factor", "up_sampling_factor"});
    if (sampling_factor_key != -1) {
        up_sampling_factor = gguf_get_val_u32(meta, sampling_factor_key);
    }
    
    int max_gen_key = search_for_gguf_keys(meta, {"parler-tts.decoder.max_generation", "max_generation"});
    if (max_gen_key != -1) {
        max_generation_size = gguf_get_val_u32(meta, max_gen_key);
    }
    
    for (int i = 0; i < (int) layers.size(); i++)  {
        std::string stride_kw = "dac_layer_stride_" + std::to_string(i);
        std::string padding_kw = "dac_layer_padding_" + std::to_string(i);
        int layer_stride_key = search_for_gguf_keys(meta, {"dac." + stride_kw, stride_kw});
        if (layer_stride_key == -1) {
            TTS_ABORT("key %s must be specified in gguf file.", ("dac." + stride_kw).c_str());
        }
        layers[i].stride = gguf_get_val_u32(meta, layer_stride_key);
        int layer_padding_key = search_for_gguf_keys(meta, {"dac." + padding_kw, padding_kw});
        if (layer_padding_key == -1) {
            TTS_ABORT("key %s must be specified in gguf file.", ("dac." + padding_kw).c_str());
        }
        layers[i].padding = gguf_get_val_u32(meta, layer_padding_key);
    }
}

void dac_model::prep_layers(gguf_context * meta) {
    for (int i = 0; i < n_heads; i++) {
        dac_quantize_layer l;
        quantizer_layers.push_back(l);
    }
    
    for (int i = 0; i < n_layers; i++) {
        dac_layer l;
        // all dac layers have 3 residual units
        for (int ii = 0; ii < 3; ii++) {
            dac_residual_unit u;
            l.residual_blocks.push_back(u);
        }
        layers.push_back(l);
    }
}
