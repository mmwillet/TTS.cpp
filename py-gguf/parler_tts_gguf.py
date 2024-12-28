import gguf
from pathlib import Path
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import torch
from enum import IntEnum
import json


class SentencePieceTokenTypes(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


DAC_RESIDUAL_UNIT_PARTS = {
    "block.0.alpha": "res.initial.alpha",
    "block.1.bias": "res.initial.bias",
    "block.1.weight": "res.initial.weight",
    "block.2.alpha": "res.final.alpha",
    "block.3.bias": "res.final.bias",
    "block.3.weight": "res.final.weight",
}

DAC_DECODER_PARTS = {
    'model.0.bias': "initial.bias",
    'model.0.weight': "initial.weight",
    'model.1': "decoder_block.1",
    'model.2': "decoder_block.2",
    'model.3': "decoder_block.3",
    'model.4': "decoder_block.4",
    "model.5.alpha": "final.alpha",
    'model.6.bias': "final.bias",
    'model.6.weight': "final.weight",
}

DAC_DECODER_BLOCK_PARTS = {
    "block.2": "residual_unit.0",
    "block.3": "residual_unit.1",
    "block.4": "residual_unit.2",
    "block.0.alpha": "final.alpha",
    "block.1.bias": "final.bias",
    "block.1.weight": "final.weight",
}


class ParlerTTSEncoder:
    def __init__(self, model_path: Path | str = "./parler-tts.gguf", encode_large: bool = False):
        self.path = model_path if isinstance(model_path, Path) else Path(model_path)
        # Configure GGUF Writer
        self.gguf_writer = gguf.GGUFWriter(path=None, arch="parler-tts")
        self._model = None
        self.shared_token_embeddings_found = False
        self.text_encoder_tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.T5ENCODER, 24)
        self._tokenizer = None
        self.encode_large = encode_large

    @property
    def model(self):
        if self._model is None:
            self._model = ParlerTTSForConditionalGeneration.from_pretrained(
                "parler-tts/parler-tts-mini-v1" if not self.encode_large else "parler-tts/parler-tts-large-v1"
            )
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                "parler-tts/parler-tts-mini-v1" if not self.encode_large else "parler-tts/parler-tts-large-v1"
            )
        return self._tokenizer

    def prepare_tensors(self, text_encoding_prompt: str):
        self.prepare_text_encoding(text_encoding_prompt)
        self.prep_audio_encoder()
        self.prep_decoder()

    def prepare_text_encoding(self, prompt: str):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        model_kwargs = {"input_ids": input_ids}
        inputs_tensor, model_input_name, model_kwargs = self.model._prepare_model_inputs(
            None, self.model.generation_config.bos_token_id, model_kwargs
        )
        self.model._prepare_special_tokens(self.model.generation_config, False, device=inputs_tensor.device)
        model_kwargs["use_cache"] = self.model.generation_config.use_cache

        model_kwargs["attention_mask"] = self.model._prepare_attention_mask_for_generation(
            inputs_tensor, self.model.generation_config._pad_token_tensor, self.model.generation_config._eos_token_tensor
        )
        # encoder_outputs are created and added to `model_kwargs`
        model_kwargs = self.model._prepare_text_encoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, self.model.generation_config
        )
        data = model_kwargs["encoder_outputs"].last_hidden_state.to(dtype=torch.float32).detach().squeeze().numpy()
        self.gguf_writer.add_tensor("decoder.text_encoding", data, raw_dtype=gguf.GGMLQuantizationType.F32)
        self.gguf_writer.add_uint32(f"parler-tts.decoder.encode_length", data.shape[0])

    def prep_audio_encoder(self):
        modules = {name: module for name, module in self.model.audio_encoder.model.decoder.named_modules()}
        for name, param in self.model.audio_encoder.model.decoder.named_parameters():
            parts_for_normalized_weights = name.split(".")
            if parts_for_normalized_weights[-1] == "weight_g":
                for hook in modules[".".join(parts_for_normalized_weights[:-1])]._forward_pre_hooks.values():
                    hook(modules[".".join(parts_for_normalized_weights[:-1])], None)
                # let's get the actual weight
                param = modules[".".join(parts_for_normalized_weights[:-1])].weight
                parts_for_normalized_weights[-1] = "weight"
                name = ".".join(parts_for_normalized_weights)
            elif parts_for_normalized_weights[-1] == "weight_v":
                # ignore
                continue
            parts = name.split(".block")
            new_name = ["audio_encoder"]
            for i, part in enumerate(parts):
                part = f"block{part}" if i > 0 else part
                if i == 0:
                    if part not in DAC_DECODER_PARTS:
                        raise ValueError(f"Part {part} is not in DAC_ENCODER_PARTS.")
                    new_name.append(DAC_DECODER_PARTS[part])
                elif i == 1:
                    if part not in DAC_DECODER_BLOCK_PARTS:
                        raise ValueError(f"Part {part} is not in DAC_ENCODER_BLOCK_PARTS.")
                    new_name.append(DAC_DECODER_BLOCK_PARTS[part])
                elif i == 2:
                    if part not in DAC_RESIDUAL_UNIT_PARTS:
                        raise ValueError(f"Part {part} is not in DAC_RESIDUAL_UNIT_PARTS.")
                    new_name.append(DAC_RESIDUAL_UNIT_PARTS[part])
                else:
                    raise ValueError(f"WTF!? here are parts {parts}")
            new_name = ".".join(new_name)
            data = param.data.to(dtype=torch.float32)
            data = data.numpy()
            self.gguf_writer.add_tensor(new_name, data, raw_dtype=gguf.GGMLQuantizationType.F32)

        modules = {name: module for name, module in self.model.audio_encoder.model.quantizer.named_modules()}
        for name, param in self.model.audio_encoder.model.quantizer.named_parameters():
            if "in_proj" in name:
                # the input projection for the quantized layers is only used when encoding audio not decoding.
                continue
            parts_for_normalized_weights = name.split(".")
            if parts_for_normalized_weights[-1] == "weight_g":
                for hook in modules[".".join(parts_for_normalized_weights[:-1])]._forward_pre_hooks.values():
                    hook(modules[".".join(parts_for_normalized_weights[:-1])], None)
                # let's get the actual weight
                param = modules[".".join(parts_for_normalized_weights[:-1])].weight
                parts_for_normalized_weights[-1] = "weight"
                name = ".".join(parts_for_normalized_weights)
            elif parts_for_normalized_weights[-1] == "weight_v":
                # ignore because we will encode the weight when we see the weight_g param
                continue
            new_name = f"audio_encoder.{name}"
            data = param.data.to(dtype=torch.float32).numpy()
            self.gguf_writer.add_tensor(new_name, data, raw_dtype=gguf.GGMLQuantizationType.F32)

    def prep_decoder(self):
        prompt_emb = self.model.embed_prompts.weight.data.to(dtype=torch.float32).numpy()
        self.gguf_writer.add_tensor("decoder.embed_prompts", prompt_emb, raw_dtype=gguf.GGMLQuantizationType.F32)
        positional_embed = self.model.decoder.model.decoder.embed_positions.weights.to(dtype=torch.float32).numpy()
        self.gguf_writer.add_tensor("decoder.positional_embed", positional_embed, raw_dtype=gguf.GGMLQuantizationType.F32)
        for name, param in self.model.decoder.model.decoder.named_parameters():
            new_name = f"decoder.{name}"
            data = param.data.to(dtype=torch.float32).squeeze().numpy()
            self.gguf_writer.add_tensor(new_name, data, raw_dtype=gguf.GGMLQuantizationType.F32)
        for name, param in self.model.decoder.lm_heads.named_parameters():
            new_name = f"decoder.lm_heads.{name}.head"
            data = param.data.to(dtype=torch.float32)
            if len(data.shape) > 1:
                data = data.squeeze().numpy()
            else:
                data = data.numpy()
            self.gguf_writer.add_tensor(new_name, data, raw_dtype=gguf.GGMLQuantizationType.F32)

    def prepare_metadata(self):
        total_params, shared_params, expert_params, expert_count = self.gguf_writer.get_total_parameter_count()
        self.metadata = gguf.Metadata.load(None, None, "parler-tts-mini-v1" if not self.encode_large else "parler-tts-large-v1", total_params)

        # Fallback to model directory name if metadata name is still missing
        if self.metadata.name is None:
            self.metadata.name = "parler-tts-mini-v1" if not self.encode_large else "parler-tts-large-v1"

        # Generate parameter weight class (useful for leader boards) if not yet determined
        if self.metadata.size_label is None and total_params > 0:
            self.metadata.size_label = gguf.size_label(total_params, shared_params, expert_params, expert_count)

        # Extract the encoding scheme from the file type name. e.g. 'gguf.LlamaFileType.MOSTLY_Q8_0' --> 'Q8_0'
        output_type: str = gguf.LlamaFileType.ALL_F32.name.partition("_")[2]

        # Filename Output
        self.path = self.path.parent / gguf.fill_templated_filename(self.path.name, output_type)
        self.set_type()
        self.metadata.set_gguf_meta_model(self.gguf_writer)
        self.set_gguf_parameters()
        self.set_vocab()
        self.gguf_writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    def set_gguf_parameters(self):
        # basic stuff
        self.gguf_writer.add_pad_token_id(self.model.config.pad_token_id)
        self.gguf_writer.add_decoder_start_token_id(self.model.config.decoder_start_token_id)

        # audio encoder config
        # this is essentially hard coded for the parler version of dac
        self.gguf_writer.add_uint32("dac.up_scaling_factor", 512)
        for i in range(4):
            self.gguf_writer.add_uint32(f"dac.dac_layer_stride_{i}", self.model.audio_encoder.model.decoder.model[i+1].block[1].stride[0])
            self.gguf_writer.add_uint32(f"dac.dac_layer_padding_{i}", self.model.audio_encoder.model.decoder.model[i+1].block[1].padding[0])

        # audio token config
        self.gguf_writer.add_uint32(f"audio.bos_token_id", self.model.decoder.config.bos_token_id)
        self.gguf_writer.add_uint32(f"audio.eos_token_id", self.model.decoder.config.eos_token_id)

        # decoder config
        self.hparams = self.model.config.decoder.to_dict()
        self.audio_hparams = self.model.config.audio_encoder.to_dict()
        self.gguf_writer.arch = "parler-tts.decoder"
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.hidden_size", self.hparams["hidden_size"])
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.output_heads", self.hparams["num_codebooks"])
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.max_generation", self.model.generation_config.max_length)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.out_vocab_size", self.hparams["vocab_size"])
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.audio_vocab_size", self.audio_hparams["codebook_size"])
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.num_hidden_layers", self.hparams["num_hidden_layers"])

        self.gguf_writer.arch = "parler-tts"
        self.gguf_writer.add_file_type(gguf.LlamaFileType.ALL_F32)

    def set_type(self):
        self.gguf_writer.add_type(gguf.GGUFType.MODEL)

    def set_vocab(self):
        vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        ordered_vocab = [vocab[i].replace('▁', " ") for i in range(max(vocab.keys()) + 1)]
        scores_by_token = {token: score for (token, score) in json.loads(self.tokenizer._tokenizer.to_str())['model']['vocab']}
        scores = [scores_by_token[vocab[i]] for i in range(max(vocab.keys()) + 1)]
        self.gguf_writer.add_token_list(ordered_vocab)
        self.gguf_writer.add_token_scores(scores)
        # these are hardcoded for all parler tts models at the moment.
        self.gguf_writer.add_eos_token_id(1)
        self.gguf_writer.add_unk_token_id(2)
        self.gguf_writer.add_add_bos_token(False)
        self.gguf_writer.add_add_eos_token(True)

    def write(self, text_encoding_prompt: str):
        self.prepare_tensors(text_encoding_prompt)
        self.prepare_metadata()
        self.gguf_writer.write_header_to_file(path=self.path)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file(progress=True)
        self.gguf_writer.close()
