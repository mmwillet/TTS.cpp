import gguf
from pathlib import Path
from transformers import AutoTokenizer
from tokenizers.models import Unigram
from parler_tts import ParlerTTSForConditionalGeneration
import torch
import json
from .tts_encoder import TTSEncoder
from .tensor_util import get_regularized_weight

# DAC_RESIDUAL_UNIT_PARTS, DAC_DECODER_PARTS, DAC_DECODER_BLOCK_PARTS are static mappings
# of the pytorch DAC Model parameter names to easily interpretable TTS.cpp names (saved to the
# GGUF file).
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

# The default repositories from which to pull Parler TTS torch models.
DEFAULT_PARLER_REPO_MINI_ID = "parler-tts/parler-tts-mini-v1"
DEFAULT_PARLER_REPO_LARGE_ID = "parler-tts/parler-tts-large-v1"

# The architecture string used by TTS.cpp to assign the appropriate TTS Model and Runner.
PARLER_TTS_ARCHITECTURE = "parler-tts"

# The default conditional prompt from which to generate the voice for the encoded Parler TTS model
DEFAULT_CONDITIONAL_PROMPT = "female voice"


class ParlerTTSEncoder(TTSEncoder):
    """
    The purpose of this class is to encode and write the tensors and model configuration for the  Parler-TTS generative
    Decoder and Audio Encoder models into a GGUF file for TTS.cpp. Only the tensors necessary to Generation are handled
    by this class; specifically Parler's T5-encoder for condition prompt cross attention is separately managed by
    the T5Encoder class.

    General usage:

    ```python
    from tts_encoders import ParlerTTSEncoder, DEFAULT_PARLER_REPO_MINI_ID

    gguf_encoder = ParlerTTSEncoder("some/local/path.gguf", DEFAULT_PARLER_REPO_MINI_ID, "scratchy male voice")
    gguf_encoder.write()
    ```
    """
    def __init__(self, model_path: Path | str = "./parler-tts.gguf", repo_id: Path | str = DEFAULT_PARLER_REPO_MINI_ID,
                 text_encoding_prompt: str = DEFAULT_CONDITIONAL_PROMPT):
        """
        :param Path or str model_path: the path to save the GGUF file.
        :param Path or str repo_id: the hugging face repo or local path from which to load the pytorch Parler-TTS model.
        """
        super().__init__(model_path = model_path, architecture = PARLER_TTS_ARCHITECTURE)
        self.text_encoding_prompt = text_encoding_prompt
        self.repo_id = repo_id
        self._tokenizer = None
        self._model = None

    @property
    def model(self) -> ParlerTTSForConditionalGeneration:
        if self._model is None:
            try:
                self._model = ParlerTTSForConditionalGeneration.from_pretrained(self.repo_id)
            except Exception as e:
                self.logger.exception(
                    f"Failed with exception, {e}, when attempting to obtain ParlerTTSForConditionalGeneration at path or repo: '{self.repo_id}'"
                )
                raise e
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
            except Exception as e:
                self.logger.exception(
                    f"Failed with exception, {e}, when attempting to obtain obtain tokenizer at path or repo: '{self.repo_id}'"
                )
                raise e
        return self._tokenizer

    def prepare_tensors(self):
        """
        Implementation of TTSEncoder's Abstract method see TTSEncoder for more information
        """
        self.prepare_text_encoding_tensors(self.text_encoding_prompt)
        self.prepare_audio_encoder_tensors()
        self.prepare_decoder_tensors()

    def prepare_text_encoding_tensors(self, prompt: str):
        """
        Prepares the conditional prompt for Parler TTS generation as a precomputed tensor and
        adds it to the GGUF file writer.

        :param str prompt: the conditional prompt to use to define the TTS voice. This should be a short
            description of how the voice should sound.
        """
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
        data = model_kwargs["encoder_outputs"].last_hidden_state
        self.set_tensor("decoder.text_encoding", data)
        self.gguf_writer.add_uint32(f"parler-tts.decoder.encode_length", data.shape[0])

    def prepare_audio_encoder_tensors(self):
        """
        Prepares and writes the tensors for the DAC audio encoder model used post-generation to conform Parler-TTS
        into audio format to the GGUF file writer.
        """
        modules = {name: module for name, module in self.model.audio_encoder.model.decoder.named_modules()}
        for name, param in self.model.audio_encoder.model.decoder.named_parameters():
            name_parts = name.split(".")
            if name_parts[-1] == "weight_g":
                param = get_regularized_weight(modules, name)
                name_parts[-1] = "weight"
                name = ".".join(name_parts)
            elif name_parts[-1] == "weight_v":
                # ignore because we will encode the weight when we see the weight_g param
                continue
            parts = name.split(".block")
            new_name = ["audio_encoder"]
            for i, part in enumerate(parts):
                part = f"block{part}" if i > 0 else part
                if i == 0:
                    if part not in DAC_DECODER_PARTS:
                        self.logger.exception(f"Found unexpected tensor in DAC model, '{name}'.")
                        raise ValueError(f"Part {part} is not in DAC_ENCODER_PARTS.")
                    new_name.append(DAC_DECODER_PARTS[part])
                elif i == 1:
                    if part not in DAC_DECODER_BLOCK_PARTS:
                        self.logger.exception(f"Found unexpected tensor in DAC model, '{name}'.")
                        raise ValueError(f"Part {part} is not in DAC_ENCODER_BLOCK_PARTS.")
                    new_name.append(DAC_DECODER_BLOCK_PARTS[part])
                elif i == 2:
                    if part not in DAC_RESIDUAL_UNIT_PARTS:
                        self.logger.exception(f"Found unexpected tensor in DAC model, '{name}'.")
                        raise ValueError(f"Part {part} is not in DAC_RESIDUAL_UNIT_PARTS.")
                    new_name.append(DAC_RESIDUAL_UNIT_PARTS[part])
                else:
                    self.logger.exception(f"Found unexpected tensor in DAC model, '{name}'.")
                    raise ValueError(f"DAC tensor '{name}' cannot be interpreted or encoded by {self.__class__}.")
            new_name = ".".join(new_name)
            self.set_tensor(new_name, param)

        modules = {name: module for name, module in self.model.audio_encoder.model.quantizer.named_modules()}
        for name, param in self.model.audio_encoder.model.quantizer.named_parameters():
            if "in_proj" in name:
                # the input projection for the quantized layers is only used when encoding audio not decoding.
                continue
            name_parts = name.split(".")
            if name_parts[-1] == "weight_g":
                param = get_regularized_weight(modules, name)
                name_parts[-1] = "weight"
                name = ".".join(name_parts)
            elif name_parts[-1] == "weight_v":
                # ignore because we will encode the weight when we see the weight_g param
                continue
            new_name = f"audio_encoder.{name}"
            self.set_tensor(new_name, param)

    def prepare_decoder_tensors(self):
        """
        Prepares and writes the tensors for the Parler-TTS cross attentional decoder to the GGUF file writer.
        """
        prompt_emb = self.model.embed_prompts.weight
        self.set_tensor("decoder.embed_prompts", prompt_emb)
        positional_embed = self.model.decoder.model.decoder.embed_positions.weights
        self.set_tensor("decoder.positional_embed", positional_embed)
        for name, param in self.model.decoder.model.decoder.named_parameters():
            new_name = f"decoder.{name}"
            self.set_tensor(new_name, param)
        for name, param in self.model.decoder.lm_heads.named_parameters():
            new_name = f"decoder.lm_heads.{name}.head"
            data = param.data.to(dtype=torch.float32)
            if len(data.shape) > 1:
                data = data.squeeze().numpy()
            else:
                data = data.numpy()
            self.set_tensor(new_name, data)

    def prepare_metadata(self):
        """
        Implementation of TTSEncoder's Abstract method see TTSEncoder for more information
        """
        total_params, shared_params, expert_params, expert_count = self.gguf_writer.get_total_parameter_count()
        self.metadata = gguf.Metadata.load(None, None, self.repo_id, total_params)

        # Generate parameter weight class (useful for leader boards) if not yet determined
        if self.metadata.size_label is None and total_params > 0:
            self.metadata.size_label = gguf.size_label(total_params, shared_params, expert_params, expert_count)

        self.set_type()
        self.set_gguf_parameters()
        self.metadata.set_gguf_meta_model(self.gguf_writer)
        self.set_vocab()
        self.gguf_writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    def set_gguf_parameters(self):
        """
        The purpose of this function is to add general model configuration to the GGUF file writer.
        """
        self.gguf_writer.add_pad_token_id(self.model.config.pad_token_id)
        self.gguf_writer.add_decoder_start_token_id(self.model.config.decoder_start_token_id)

        # ---- DAC Audio Encoder configuration ----

        # the upscaling factor represents the input to output upscale rate in the DAC Audio encoder.
        # It is static for all versions of DAC used by Parler-TTS
        self.gguf_writer.add_uint32("dac.up_scaling_factor", 512)
        for i in range(4):
            self.gguf_writer.add_uint32(f"dac.dac_layer_stride_{i}", self.model.audio_encoder.model.decoder.model[i+1].block[1].stride[0])
            self.gguf_writer.add_uint32(f"dac.dac_layer_padding_{i}", self.model.audio_encoder.model.decoder.model[i+1].block[1].padding[0])

        # DAC audio token configuration
        self.gguf_writer.add_uint32(f"audio.bos_token_id", self.model.decoder.config.bos_token_id)
        self.gguf_writer.add_uint32(f"audio.eos_token_id", self.model.decoder.config.eos_token_id)

        # ---- Parler TTS Decoder configuration ----

        # hparams and audio_hparams represent the independent model configuration for the Parler-TTS Decoder and
        # the DAC Audio Encoder models respectively. 
        hparams = self.model.config.decoder.to_dict()
        audio_hparams = self.model.config.audio_encoder.to_dict()
        # Overwrite the architecture so that context length and head count are stored appropriately under the
        # decoder context in the GGUF file.
        self.gguf_writer.arch = f"{PARLER_TTS_ARCHITECTURE}.decoder"
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.hidden_size", hparams["hidden_size"])
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.output_heads", hparams["num_codebooks"])
        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.max_generation", self.model.generation_config.max_length)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.out_vocab_size", hparams["vocab_size"])
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.audio_vocab_size", audio_hparams["codebook_size"])
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.num_hidden_layers", hparams["num_hidden_layers"])
        # reset the architecture
        self.gguf_writer.arch = PARLER_TTS_ARCHITECTURE

        # The file type setting is purely for describing the primary precision of the model as it is stored in the GGUF file.
        # This setting *does not* enforce the tensor format or alter tensor processing capabilities in TTS.cpp and is only
        # used for reporting.
        self.gguf_writer.add_file_type(gguf.LlamaFileType.ALL_F32)

    def set_type(self):
        self.gguf_writer.add_type(gguf.GGUFType.MODEL)

    def set_vocab(self):
        """
        The purpose of this function is to add the vocab and configuration for the Parler TTS unigram tokenizer
        to the GGUF file writer.
        """
        assert hasattr(self.tokenizer, "_tokenizer") \
               and hasattr(self.tokenizer._tokenizer, "model") \
               and isinstance(self.tokenizer._tokenizer.model, Unigram), f"Found non-unigram tokenizer. Currently tokenizer of type {tokenizer.__class__} is not supported."
        vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        ordered_vocab = [vocab[i].replace('▁', " ") for i in range(max(vocab.keys()) + 1)]
        scores_by_token = {token: score for (token, score) in json.loads(self.tokenizer._tokenizer.to_str())['model']['vocab']}
        scores = [scores_by_token[vocab[i]] for i in range(max(vocab.keys()) + 1)]
        self.gguf_writer.add_token_list(ordered_vocab)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_eos_token_id(self.model.config.text_encoder.eos_token_id)
        self.gguf_writer.add_unk_token_id(self.tokenizer.unk_token_id)
        self.gguf_writer.add_add_bos_token(False)
        self.gguf_writer.add_add_eos_token(True)
