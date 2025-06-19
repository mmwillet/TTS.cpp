import gguf
from pathlib import Path
import torch
from .dac_gguf_encoder import DACEncoder
from dia.model import Dia
from dia.state import EncoderInferenceState
from dia.layers import DiaModel

# The default repositories from which to pull the Dia torch model.
DEFAULT_DIA_REPO_ID = "nari-labs/Dia-1.6B"

# The architecture string used by TTS.cpp to assign the appropriate TTS Model and Runner.
DIA_ARCHITECTURE = "dia"


class DiaEncoder(DACEncoder):
    """
    The purpose of this class is to encode and write the tensors and model configuration for the Dia Text to Speech
    model into a GGUF file for TTS.cpp.

    General usage:

    ```python
    from tts_encoders import DiaEncoder, DEFAULT_DIA_REPO_ID

    gguf_encoder = DiaEncoder("some/local/path.gguf", DEFAULT_DIA_REPO_ID)
    gguf_encoder.write()
    ```
    """
    def __init__(self, model_path: Path | str = "./dia.gguf", repo_id: Path | str = DEFAULT_DIA_REPO_ID):
        """
        :param Path or str model_path: the path to save the GGUF file.
        :param Path or str repo_id: the hugging face repo or local path from which to load the pytorch Dia model.
        """
        super().__init__(model_path = model_path, architecture = DIA_ARCHITECTURE)
        self.repo_id = repo_id
        self._tokenizer = None
        self._model = None

    @property
    def model(self) -> DiaModel:
        if self._model is None:
            try:
                self._model = Dia.from_pretrained(self.repo_id, compute_dtype="float32")
            except Exception as e:
                self.logger.exception(
                    f"Failed with exception, {e}, when attempting to obtain Dia at path or repo: '{self.repo_id}'"
                )
                raise e
        return self._model.model

    @property
    def dac_model(self):
        if self._model is None:
            try:
                self._model = Dia.from_pretrained(self.repo_id, compute_dtype="float32")
            except Exception as e:
                self.logger.exception(
                    f"Failed with exception, {e}, when attempting to obtain Dia at path or repo: '{self.repo_id}'"
                )
                raise e
        return self._model.dac_model


    def prepare_tensors(self):
        """
        Implementation of TTSEncoder's Abstract method see TTSEncoder for more information
        """
        self.prepare_decoder_tensors()
        self.prepare_encoder_tensors()
        self.prepare_dac_audio_encoder_tensors()

    def prepare_decoder_tensors(self):
        """
        Prepares and writes the tensors for the Dia decoder to the GGUF file writer.
        """
        base = "dia.decoder"
        for name, param in self.model.decoder.named_parameters():
            parts = name.split(".")
            if parts[0] == "embeddings":
                self.set_tensor(f"{base}.{parts[0]}.{parts[1]}", param)
            elif parts[0] == "norm":
                self.set_tensor(f"{base}.norm", param)
            elif parts[0] == "logits_dense":
                heads = param.shape[1]
                for i in range(heads):
                    head = param.data[:, i]
                    self.set_tensor(f"{base}.heads.{i}", head.transpose(0,1))
            elif parts[0] == "layers":
                nn = f"{base}.{parts[0]}.{parts[1]}"
                if (parts[2] == "mlp" and parts[3] == "wi_fused"):
                    # the typical MLP gate and up layers are fused together in Dia's implementation, so we split them and separately encode them.
                    gate = param.data[:, 0]
                    up = param.data[:, 1]
                    self.set_tensor(f"{nn}.gate", gate.transpose(0,1))
                    self.set_tensor(f"{nn}.up", up.transpose(0,1))
                elif parts[2] == "mlp":
                    self.set_tensor(f"{nn}.{parts[3]}", param.data.transpose(0, 1))
                elif parts[2] == "self_attention":
                    data = param.data.reshape(param.shape[0], -1).transpose(0, 1)
                    if parts[3] == "o_proj":
                        data = param.data.reshape(-1, param.shape[-1]).transpose(0, 1)
                    self.set_tensor(f"{nn}.self_{parts[3]}", data)
                elif parts[2] == "cross_attention":
                    data = param.data.reshape(param.shape[0], -1).transpose(0, 1)
                    if parts[3] == "o_proj":
                        data = param.data.reshape(-1, param.shape[-1]).transpose(0, 1)
                    self.set_tensor(f"{nn}.cross_{parts[3]}", data)
                else:
                    self.set_tensor(f"{nn}.{parts[2]}", param)

    def prepare_encoder_tensors(self):
        """
        Prepares and writes the tensors for the Dia encoder to the GGUF file writer.
        """
        base = "dia.encoder"
        for name, param in self.model.encoder.named_parameters():
            parts = name.split(".")
            if parts[0] == "embedding":
                self.set_tensor(f"{base}.embedding", param)
            elif parts[0] == "norm":
                self.set_tensor(f"{base}.norm", param)
            elif parts[0] == "layers":
                nn = f"{base}.{parts[0]}.{parts[1]}"
                if (parts[2] == "mlp" and parts[3] == "wi_fused"):
                    # the typical MLP gate and up layers are fused together in Dia's implementation, so we split them and separately encode them.
                    gate = param.data[:, 0]
                    up = param.data[:, 1]
                    self.set_tensor(f"{nn}.gate", gate.transpose(0,1))
                    self.set_tensor(f"{nn}.up", up.transpose(0,1))
                elif parts[2] == "mlp":
                    self.set_tensor(f"{nn}.{parts[3]}", param.data.transpose(0, 1))
                elif parts[2] == "self_attention":
                    data = param.data.reshape(param.shape[0], -1).transpose(0, 1)
                    if parts[3] == "o_proj":
                        data = param.data.reshape(-1, param.shape[-1]).transpose(0,1)
                    self.set_tensor(f"{nn}.{parts[3]}", data)
                else:
                    self.set_tensor(f"{nn}.{parts[2]}", param)


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
        self.gguf_writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    def set_gguf_parameters(self):
        """
        The purpose of this function is to add general model configuration to the GGUF file writer.
        """
        self.set_dac_config(self.model.config.data.audio_bos_value, self.model.config.data.audio_eos_value);

        # ---- Dia Configuration ----
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.attn_head_size", self.model.config.model.encoder.head_dim)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.eos_token_id", self.model.config.data.audio_eos_value)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.bos_token_id", self.model.config.data.audio_bos_value)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.pad_token_id", self.model.config.data.audio_pad_value)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.max_delay", max(self.model.config.data.delay_pattern))


        # ---- Dia Encoder Configuration ----
        # Overwrite the architecture so that encoder config is stored appropriately under the encoder context in the GGUF file.
        self.gguf_writer.arch = f"{DIA_ARCHITECTURE}.encoder"
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.max_context_length", self.model.config.data.text_length)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.attn_heads", self.model.config.model.encoder.n_head)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.layers", self.model.config.model.encoder.n_layer)

        # ---- Dia Decoder Configuration ----
        # Overwrite the architecture so that encoder config is stored appropriately under the decoder context in the GGUF file.
        self.gguf_writer.arch = f"{DIA_ARCHITECTURE}.decoder"
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.hidden_size", self.model.config.model.decoder.n_embd)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.layers", self.model.config.model.decoder.n_layer)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.output_heads", self.model.config.data.channels)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.attn_heads", self.model.config.model.decoder.gqa_query_heads)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.query_heads", self.model.config.model.decoder.kv_heads)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.output_vocab_size", self.model.config.model.tgt_vocab_size)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.audio_vocab_size", self.model.config.data.audio_eos_value)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.max_generation_size", self.model.config.data.audio_length)

        # Reset the architecture on the gguf writer
        self.gguf_writer.arch = DIA_ARCHITECTURE

        # The file type setting is purely for describing the primary precision of the model as it is stored in the GGUF file.
        # This setting *does not* enforce the tensor format or alter tensor processing capabilities in TTS.cpp and is only
        # used for reporting.
        self.gguf_writer.add_file_type(gguf.LlamaFileType.ALL_F32)

    def set_type(self):
        self.gguf_writer.add_type(gguf.GGUFType.MODEL)
