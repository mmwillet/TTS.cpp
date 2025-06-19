from huggingface_hub import hf_hub_download
from pathlib import Path
from snac import SNAC
from snac.layers import DecoderBlock
from transformers import AutoModelForCausalLM
from transformers.models.llama import LlamaForCausalLM
from typing import Dict
from .dac_gguf_encoder import DAC_RESIDUAL_UNIT_PARTS
from .tts_encoder import TTSEncoder
from .tensor_util import get_normalized_weight_from_parametrizations

import gguf
import json
import math
import torch

DEFAULT_ORPHEUS_REPO_ID = "canopylabs/orpheus-3b-0.1-ft"
DEFAULT_SNAC_REPO_ID = "hubertsiuzdak/snac_24khz"
ORPHEUS_ARCHITECTURE = "orpheus"


class OrpheusEncoder(TTSEncoder):
    """
    The purpose of this class is to encode and write the tensors and model configuration for the Orpheus TTS model that
    into a GGUF file.

    General Usage:

    ```python
    from tts_encoders import OrpheusEncoder

    gguf_encoder = OrpheusEncoder("some/local/path.gguf")
    gguf_encoder.write()
    ```
    """
    def __init__(self, model_path: Path | str = "./orpheus.gguf", repo_id: Path | str = DEFAULT_ORPHEUS_REPO_ID,
                 snac_repo_id: Path | str = DEFAULT_SNAC_REPO_ID):
        """
        :param Path or str model_path: The path to save the generated GGUF file.
        :param Path or str repo_id: The path or repository from which to pull the orpheus model and its tokenizer.
        :param Path or str snac_repo_id: The path or repository from which to pull the SNAC audio decoder.
        """
        super().__init__(model_path=model_path, architecture=ORPHEUS_ARCHITECTURE)
        self._model = None
        self._snac_model = None
        self._tokenizer_json = None
        self._config = None
        self.repo_id = repo_id
        self.snac_repo_id = snac_repo_id

    @property
    def model(self) -> LlamaForCausalLM:
        if self._model is None:
            try:
                self._model = AutoModelForCausalLM.from_pretrained(self.repo_id).eval().to(device="cpu")
            except Exception as e:
                self.logger.exception(
                    f"Failed with exception, {e}, when attempting to obtain Orpheus at path or repo: '{self.repo_id}'"
                )
                raise e
        return self._model

    @property
    def snac_model(self) -> SNAC:
        if self._snac_model is None:
            try:
                self._snac_model = SNAC.from_pretrained(self.snac_repo_id).eval().to("cpu")
            except Exception as e:
                self.logger.exception(
                    f"Failed with exception, {e}, when attempting to obtain SNAC Model at path or repo: '{self.snac_repo_id}'"
                )
                raise e
        return self._snac_model

    @property
    def tokenizer_json(self) -> Dict:
        if self._tokenizer_json is None:
            try:
                conf_path = hf_hub_download(repo_id=self.repo_id, filename='tokenizer.json')
            except Exception as e:
                self.logger.exception(
                    f"Failed with exception, {e}, attempting to obtain tokenizer.json via repository '{self.repo_id}'."
                )
                raise e
            with open(conf_path, "r+") as f:
                self._tokenizer_json = json.load(f)
        return self._tokenizer_json

    def simplify_snac_name(self, name: str) -> str:
        parts = name.split(".")
        model_index = int(parts[0])
        if model_index == 6:
            return "alpha_out"
        elif model_index == 7:
            return f"final.{parts[1]}"
        elif model_index == 0:
            return f"in.{parts[1]}"
        elif model_index == 1:
            return f"up.{parts[1]}"
        else:
            model_index -= 2
            layer_index = int(parts[2])
            if layer_index == 0:
                return f"layers.{model_index}.alpha"
            elif layer_index == 1:
                return f"layers.{model_index}.{parts[-1]}"
            elif layer_index == 2:
                return f"layers.{model_index}.noise_{parts[-1]}"
            else:
                base = f"layers.{model_index}.residual_unit.{layer_index - 3}"
                return base + "." + DAC_RESIDUAL_UNIT_PARTS[".".join(parts[-3:])]

    def prepare_tensors(self):
        self.prepare_orpheus_tensors()
        self.prepare_snac_tensors()
        self.prepare_rope_frequencies()

    def prepare_orpheus_tensors(self):
        for name, param in self.model.model.named_parameters():
            name = f"orpheus.{name[:-7]}" # all names end in ".weight" for Orpheus
            self.set_tensor(name, param)
        self.set_tensor("orpheus.lm_head", self.model.lm_head.weight)

    def prepare_snac_tensors(self):
        modules = {n: v for n, v in self.snac_model.quantizer.named_modules()}
        for name, param in self.snac_model.quantizer.named_parameters():
            if "parametrizations.weight.original0" in name:
                param = get_normalized_weight_from_parametrizations(modules, name)
                name = name.replace("parametrizations.weight.original0", "weight")
            elif "parametrizations.weight" in name:
                continue
            self.set_tensor(f"snac.{name}", param)

        modules = {n: v for n, v in self.snac_model.decoder.model.named_modules()}
        for name, param in self.snac_model.decoder.model.named_parameters():
            if "parametrizations.weight.original0" in name:
                param = get_normalized_weight_from_parametrizations(modules, name)
                name = name.replace("parametrizations.weight.original0", "weight")
            elif "parametrizations.weight" in name:
                continue
            name = self.simplify_snac_name(name)
            self.set_tensor(f"snac.{name}", param)

    def prepare_rope_frequencies(self):
        """
        Because Llama-3 like Rotary Positional Embeddings are not currently supported out-of-the-box in GGML, 
        we need to encode the rope frequency vectors to use directly.
        """
        base = self.model.config.rope_theta
        dim = self.model.config.head_dim
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        factor = self.model.config.rope_scaling.get("factor", 8.0)
        low_freq_factor = self.model.config.rope_scaling.get("low_freq_factor", 1.0)
        high_freq_factor = self.model.config.rope_scaling.get("high_freq_factor", 4.0)
        old_context_len = self.model.config.rope_scaling.get("original_max_position_embeddings", 8192)

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        assert low_freq_wavelen != high_freq_wavelen

        rope_factors = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                rope_factors.append(1)
            elif wavelen > low_freq_wavelen:
                rope_factors.append(factor)
            else:
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                        high_freq_factor - low_freq_factor)
                rope_factors.append(1 / ((1 - smooth) / factor + smooth))

        self.set_tensor("orpheus.rope_frequencies", torch.tensor(rope_factors, dtype=torch.float32))

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

        # this is not set in Orpheus configuration or on the class level. It is passed as a
        # a default parameter to the generation function.
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.stopping_token_id", 128258)

        # ---- Orpheus configuration ----
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.hidden_size", self.model.config.hidden_size)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.vocab_size", self.model.config.vocab_size)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.attn_heads", self.model.config.num_attention_heads)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.kv_attn_heads", self.model.config.num_key_value_heads)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.head_dim", self.model.config.head_dim)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.layers", self.model.config.num_hidden_layers)
        self.gguf_writer.add_uint32(
            f"{self.gguf_writer.arch}.kv_hidden_size",
            self.model.config.hidden_size // (self.model.config.num_attention_heads // self.model.config.num_key_value_heads)
        )

        # ---- SNAC configuration ----
        self.gguf_writer.add_uint32("snac.audio_token_channels", self.snac_model.quantizer.n_codebooks)
        layer_index = 0
        for module in self.snac_model.decoder.model:
            if isinstance(module, DecoderBlock):
                self.gguf_writer.add_uint32(f"snac.snac_layer_stride_{layer_index}", module.block[1].stride[0])
                self.gguf_writer.add_uint32(f"snac.snac_layer_padding_{layer_index}", module.block[1].padding[0])
                self.gguf_writer.add_uint32(f"snac.snac_layer_grouping_{layer_index}", module.block[3].block[1].groups)
                layer_index += 1

        # The file type setting is purely for describing the primary precision of the model as it is stored in the GGUF file.
        # This setting *does not* enforce the tensor format or alter tensor processing capabilities in TTS.cpp and is only
        # used for reporting.
        self.gguf_writer.add_file_type(gguf.LlamaFileType.ALL_F32)

    def set_type(self):
        self.gguf_writer.add_type(gguf.GGUFType.MODEL)

    def set_vocab(self):
        """
        The purpose of this function is to add the vocab, merges, and configuration for Orpheus' BPE tokenizer
        to the GGUF file writer.
        """
        assert "model" in self.tokenizer_json and "type" in self.tokenizer_json["model"] and self.tokenizer_json["model"]["type"] == "BPE" \
               and "merges" in self.tokenizer_json["model"] and "vocab" in self.tokenizer_json["model"]
        tokens = list(self.tokenizer_json["model"]["vocab"].keys())
        print(f"HERE WTF is going on {len(tokens)}.")
        merges = [" ".join(pair) for pair in self.tokenizer_json["model"]["merges"]]
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_merges(merges)
        self.gguf_writer.add_eos_token_id(self.model.config.eos_token_id)
        self.gguf_writer.add_bos_token_id(self.model.config.bos_token_id)
