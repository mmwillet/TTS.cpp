#include "parler_gguf.h"

// loading the vocab to the tokenizer from gguf file.
unigram_tokenizer * tokenizer_from_gguf(gguf_context * meta) {
    std::unordered_map<std::string, uint32_t> vocab;
    std::vector<float> scores;
    int vocab_key = gguf_find_key(meta, "tokenizer.ggml.tokens");
    int vocab_size = gguf_get_arr_n(meta, vocab_key);
    scores.reserve(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        std::string val = gguf_get_arr_str(meta, vocab_key, i);
        vocab[val] = (uint32_t) i;
    }
    int scores_key = gguf_find_key(meta, "tokenizer.ggml.scores");
    int scores_size = gguf_get_arr_n(meta, scores_key);
    assert(scores_size == vocab_size);
    float * data = (float*) gguf_get_arr_data(meta, scores_key);
    for (int i = 0; i < scores_size; i++) {
        scores.push_back(data[i]);
    }
    int unkown_token_key = gguf_find_key(meta, "tokenizer.ggml.unknown_token_id");
    uint32_t token = gguf_get_val_u32(meta, unkown_token_key);
    return new unigram_tokenizer(vocab, token, scores[token], scores);
}

// Simple helper function for getting layer count from tensor name
static std::pair<int, std::string> parse_layer_count(std::string name) {
    bool found = false;
    bool after_layer = false;
    std::string digit_chars = "";
    std::string after_layer_name = "";
    for (char& c : name) {
        if (after_layer) {
            after_layer_name += c;
        } else if (std::isdigit(c)) {
            found = true;
            digit_chars += c;
        } else if (!found) {
            
        } else {
            after_layer = true;
            after_layer_name += c;
        }
    }
    return std::make_pair(std::stoi(digit_chars), after_layer_name);
}

// loading DAC model from gguf file.
static const std::map<std::string, dac_tensor> DAC_TENSOR_GGUF_LOOKUP = {
    {"initial.bias", DAC_ENCODER_IN_BIAS},
    {"initial.weight", DAC_ENCODER_IN_KERNEL},
    {"final.bias", DAC_ENCODER_OUT_BIAS},
    {"final.weight", DAC_ENCODER_OUT_KERNEL},
    {"final.alpha", DAC_ENCODER_SNAKE_ALPHA},
    {".final.alpha", DAC_ENCODER_LAYER_SNAKE_ALPHA},
    {".final.bias", DAC_ENCODER_LAYER_OUT_BIAS},
    {".final.weight", DAC_ENCODER_LAYER_OUT_KERNEL},
    {".res.initial.alpha", DAC_ENCODER_LAYER_RES_BLK_IN_SNAKE},
    {".res.initial.bias", DAC_ENCODER_LAYER_RES_BLK_IN_BIAS},
    {".res.initial.weigh", DAC_ENCODER_LAYER_RES_BLK_IN_KERNEL},
    {".res.final.alpha", DAC_ENCODER_LAYER_RES_BLK_OUT_SNAKE},
    {".res.final.bias", DAC_ENCODER_LAYER_RES_BLK_OUT_BIAS},
    {".res.final.weight", DAC_ENCODER_LAYER_RES_BLK_OUT_KERNEL},
    {".in_proj.bias", DAC_QUANTIZER_LAYER_IN_BIAS},
    {".in_proj.weight", DAC_QUANTIZER_LAYER_IN_KERNEL},
    {".out_proj.bias", DAC_QUANTIZER_LAYER_OUT_BIAS},
    {".out_proj.weight", DAC_QUANTIZER_LAYER_OUT_KERNEL},
    {".codebook.weight", DAC_QUANTIZER_LAYER_CODEBOOK},
};

void assign_residual_unit(dac_model & model, dac_residual_unit & l, std::string name, ggml_tensor * tensor) {
    dac_tensor tensor_type = DAC_TENSOR_GGUF_LOOKUP.at(name);
    switch (tensor_type) {
        case DAC_ENCODER_LAYER_RES_BLK_IN_SNAKE:
            l.in_snake_alpha = ggml_dup_tensor(model.ctx, tensor);
            model.set_tensor(l.in_snake_alpha, tensor);
            break;
        case DAC_ENCODER_LAYER_RES_BLK_OUT_SNAKE:
            l.out_snake_alpha = ggml_dup_tensor(model.ctx, tensor);
            model.set_tensor(l.out_snake_alpha, tensor);
            break;
        case DAC_ENCODER_LAYER_RES_BLK_IN_KERNEL:
            l.in_conv_kernel = ggml_dup_tensor(model.ctx, tensor);
            model.set_tensor(l.in_conv_kernel, tensor);
            break;
        case DAC_ENCODER_LAYER_RES_BLK_OUT_KERNEL:
            l.out_conv_kernel = ggml_dup_tensor(model.ctx, tensor);
            model.set_tensor(l.out_conv_kernel, tensor);
            break;
        case DAC_ENCODER_LAYER_RES_BLK_IN_BIAS:
            l.in_conv_bias = ggml_dup_tensor(model.ctx, ggml_transpose(model.ctx, tensor));
            model.set_tensor(l.in_conv_bias, tensor);
            break;
        case DAC_ENCODER_LAYER_RES_BLK_OUT_BIAS:
            l.out_conv_bias = ggml_dup_tensor(model.ctx, ggml_transpose(model.ctx, tensor));
            model.set_tensor(l.out_conv_bias, tensor);
            break;
        default:
            break;
    }
}

void assign_dac_layer(dac_model & model, dac_layer & layer, std::string name, ggml_tensor * tensor) {
    if (DAC_TENSOR_GGUF_LOOKUP.find(name) != DAC_TENSOR_GGUF_LOOKUP.end()) {
        switch(DAC_TENSOR_GGUF_LOOKUP.at(name)) {
            case DAC_ENCODER_LAYER_SNAKE_ALPHA:
                layer.snake_alpha_in = ggml_dup_tensor(model.ctx, tensor);
                model.set_tensor(layer.snake_alpha_in, tensor);
                break;
            case DAC_ENCODER_LAYER_OUT_KERNEL:
                layer.out_conv_kernel = ggml_dup_tensor(model.ctx, tensor);
                model.set_tensor(layer.out_conv_kernel, tensor);
                break;
            case DAC_ENCODER_LAYER_OUT_BIAS:
                layer.out_conv_bias = ggml_dup_tensor(model.ctx, ggml_transpose(model.ctx, tensor));
                model.set_tensor(layer.out_conv_bias, tensor);
                break;
            default:
                break;
        }
    } else if (std::find_if(name.begin(), name.end(), ::isdigit) != name.end())  {
        auto pair = parse_layer_count(name);
        int l = pair.first;
        std::string lt_name = pair.second;
        assign_residual_unit(model, layer.residual_blocks[l], lt_name, tensor);
    }
}

void assign_quantizer_layer(dac_model & model, dac_quantize_layer & layer, std::string name, ggml_tensor * tensor) {
    switch(DAC_TENSOR_GGUF_LOOKUP.at(name)) {
        case DAC_QUANTIZER_LAYER_OUT_KERNEL:
            layer.out_proj_kernel = ggml_dup_tensor(model.ctx, tensor);
            model.set_tensor(layer.out_proj_kernel, tensor);
            break;
        case DAC_QUANTIZER_LAYER_OUT_BIAS:
            layer.out_proj_bias = ggml_dup_tensor(model.ctx, ggml_transpose(model.ctx, tensor));
            model.set_tensor(layer.out_proj_bias, tensor);
            break;
        case DAC_QUANTIZER_LAYER_CODEBOOK:
            layer.codebook = ggml_dup_tensor(model.ctx, tensor);
            model.set_tensor(layer.codebook, tensor);
            break;
        default:
            break;
    }
}

void assign_to_audio_encoder(dac_model & model, std::string name, ggml_tensor * tensor) {
    model.n_tensors++;
    if (DAC_TENSOR_GGUF_LOOKUP.find(name) != DAC_TENSOR_GGUF_LOOKUP.end()) {
        switch(DAC_TENSOR_GGUF_LOOKUP.at(name)) {
            case DAC_ENCODER_IN_BIAS:
                model.in_conv_bias = ggml_dup_tensor(model.ctx, ggml_transpose(model.ctx, tensor));
                model.set_tensor(model.in_conv_bias, tensor);
                break;
            case DAC_ENCODER_IN_KERNEL:
                model.in_conv_kernel = ggml_dup_tensor(model.ctx, tensor);
                model.set_tensor(model.in_conv_kernel, tensor);
                break;
            case DAC_ENCODER_OUT_BIAS:
                model.out_conv_bias = ggml_dup_tensor(model.ctx, ggml_transpose(model.ctx, tensor));
                model.set_tensor(model.out_conv_bias, tensor);
                break;
            case DAC_ENCODER_OUT_KERNEL:
                model.out_conv_kernel = ggml_dup_tensor(model.ctx, tensor);
                model.set_tensor(model.out_conv_kernel, tensor);
                break;
            case DAC_ENCODER_SNAKE_ALPHA:
                model.snake_alpha = ggml_dup_tensor(model.ctx, tensor);
                model.set_tensor(model.snake_alpha, tensor);
                break;
            default:
                break;
        }
    } else if (std::find_if(name.begin(), name.end(), ::isdigit) != name.end())  {
        auto pair = parse_layer_count(name);
        int l = pair.first;
        std::string lt_name = pair.second;
        if (name.find("quantizers") != std::string::npos) {
            assign_quantizer_layer(model, model.quantizer_layers[l], lt_name, tensor);
        } else {
            assign_dac_layer(model, model.layers[l - 1], lt_name, tensor);
        }
    }
}

// loading Parler model from gguf file.
static const std::map<std::string, parler_tensor> PARLER_TENSOR_GGUF_LOOKUP = {
    {"layer_norm.weight", PARLER_NORM},
    {"layer_norm.bias", PARLER_NORM_BIAS},
    {"embed_prompts", PARLER_EMBD_PROMPTS},
    {"text_encoding", PARLER_TEXT_ENCODING},
    {"positional_embed", PARLER_POSITIONAL_EMBD},
    {".self_attn.q_proj.weight", PARLER_LAYER_SELF_ATTN_Q},
    {".self_attn.k_proj.weight", PARLER_LAYER_SELF_ATTN_K},
    {".self_attn.v_proj.weight", PARLER_LAYER_SELF_ATTN_V},
    {".self_attn.out_proj.weight", PARLER_LAYER_SELF_ATTN_O},
    {".self_attn_layer_norm.weight", PARLER_LAYER_SELF_ATTN_NORM},
    {".self_attn_layer_norm.bias", PARLER_LAYER_SELF_ATTN_NORM_BIAS},
    {".encoder_attn.q_proj.weight", PARLER_LAYER_ATTN_Q},
    {".encoder_attn.k_proj.weight", PARLER_LAYER_ATTN_K},
    {".encoder_attn.v_proj.weight", PARLER_LAYER_ATTN_V},
    {".encoder_attn.out_proj.weight", PARLER_LAYER_ATTN_O},
    {".encoder_attn_layer_norm.weight", PARLER_LAYER_ATTN_NORM},
    {".encoder_attn_layer_norm.bias", PARLER_LAYER_ATTN_NORM_BIAS},
    {".fc1.weight", PARLER_LAYER_FC1},
    {".fc2.weight", PARLER_LAYER_FC2},
    {".final_layer_norm.weight", PARLER_LAYER_OUT_NORM},
    {".final_layer_norm.bias", PARLER_LAYER_OUT_NORM_BIAS},
    {".weight", PARLER_EMBD},
    {".weight.head", PARLER_HEAD}
};

void assign_parler_layer(parler_tts_model * model, parler_layer * layer, std::string name, ggml_tensor * tensor) {
    switch(PARLER_TENSOR_GGUF_LOOKUP.at(name)) {
        case PARLER_LAYER_SELF_ATTN_Q:
            layer->self_attn_q_proj = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(layer->self_attn_q_proj, tensor);
            break;
        case PARLER_LAYER_SELF_ATTN_K:
            layer->self_attn_k_proj = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(layer->self_attn_k_proj, tensor);
            break;
        case PARLER_LAYER_SELF_ATTN_V:
            layer->self_attn_v_proj = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(layer->self_attn_v_proj, tensor);
            break;
        case PARLER_LAYER_SELF_ATTN_O:
            layer->self_attn_o_proj = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(layer->self_attn_o_proj, tensor);
            break;
        case PARLER_LAYER_SELF_ATTN_NORM:
            layer->self_attn_norm = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(layer->self_attn_norm, tensor);
            break;
        case PARLER_LAYER_SELF_ATTN_NORM_BIAS:
            layer->self_attn_norm_bias = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(layer->self_attn_norm_bias, tensor);
            break;
        case PARLER_LAYER_ATTN_Q:
            layer->attn_q_proj = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(layer->attn_q_proj, tensor);
            break;
        case PARLER_LAYER_ATTN_K:
            layer->attn_k_proj = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(layer->attn_k_proj, tensor);
            break;
        case PARLER_LAYER_ATTN_V:
            layer->attn_v_proj = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(layer->attn_v_proj, tensor);
            break;
        case PARLER_LAYER_ATTN_O:
            layer->attn_o_proj = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(layer->attn_o_proj, tensor);
            break;
        case PARLER_LAYER_ATTN_NORM:
            layer->attn_norm = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(layer->attn_norm, tensor);
            break;
        case PARLER_LAYER_ATTN_NORM_BIAS:
            layer->attn_norm_bias = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(layer->attn_norm_bias, tensor);
            break;
        case PARLER_LAYER_FC1:
            layer->fc1 = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(layer->fc1, tensor);
            break;
        case PARLER_LAYER_FC2:
            layer->fc2 = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(layer->fc2, tensor);
            break;
        case PARLER_LAYER_OUT_NORM:
            layer->final_norm = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(layer->final_norm, tensor);
            break;
        case PARLER_LAYER_OUT_NORM_BIAS:
            layer->final_norm_bias = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(layer->final_norm_bias, tensor);
            break;
        default:
            break;
    }
}

void assign_to_decoder(parler_tts_model * model, const std::string name, ggml_tensor * tensor) {
    model->n_tensors++;
    if (PARLER_TENSOR_GGUF_LOOKUP.find(name) != PARLER_TENSOR_GGUF_LOOKUP.end()) {
        switch (PARLER_TENSOR_GGUF_LOOKUP.at(name)) {
            case PARLER_NORM:
                model->layer_norm = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->layer_norm, tensor);
                break;
            case PARLER_NORM_BIAS:
                model->layer_norm_bias = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->layer_norm_bias, tensor);
                break;
            case PARLER_EMBD_PROMPTS:
                model->prompt_embd = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->prompt_embd, tensor);
                break;
            case PARLER_TEXT_ENCODING:
                model->precomputed_input_emb = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->precomputed_input_emb, tensor);
                break;
            case PARLER_POSITIONAL_EMBD:
                model->precomputed_positional_embds = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->precomputed_positional_embds, tensor);
                break;
            default:
                break;
        }
    } else if (std::find_if(name.begin(), name.end(), ::isdigit) != name.end())  {
        auto pair = parse_layer_count(name);
        int layer = pair.first;
        std::string lt_name = pair.second;
        if (name.find("embed_tokens") != std::string::npos) {
            model->embds[layer] = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(model->embds[layer], tensor);
        } else if (name.find("lm_heads") != std::string::npos) {
            model->heads[layer] = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(model->heads[layer], tensor);
        } else {
            assign_parler_layer(model, model->layers[layer], lt_name, tensor);
        }
    }
}

// TODO: change the load pattern for model tensors
// The configuration lookup pattern desribed here is quite bulky to support and add on to and should be replaced by something more modular.
// Additionally allocating weights to the model buffers on the first iteration through the context is unideal, as it means that the models
// must be able to predetermine their buffer size. Furthermore, this approach relies on a custom set_tensor function on the model structs;
// While this seems to work completely fine in practice it is inconsistent with llama.cpp tensor assignment patterns.
void assign_weight(parler_tts_model * model, dac_model & audio_model, std::string name, ggml_tensor * tensor) {
    std::string::size_type pos = name.find(".", 0);
    std::string top_level(name.substr(0, pos));
    std::string value(name.substr(pos + 1));
    if (tensor->data == NULL) {
        return;
    }
    if (top_level == "audio_encoder") {
        assign_to_audio_encoder(audio_model, value, tensor);
    } else if (top_level == "decoder") {
        assign_to_decoder(model, value, tensor);
    } else {
        return;
    }
}
