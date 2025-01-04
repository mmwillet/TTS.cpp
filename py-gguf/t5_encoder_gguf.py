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


class T5Encoder:
    """
    The T5 Encoder model which Parler TTS uses is a modified version of Google's T5-Flan model.
    This protocol is built to mimic the end configuration of a gguf T5 encoder encoded via llama.cpp, but
    the end model *is* different from a standard T5Encoder model. 
    """
    def __init__(self, model_path: Path | str = "./t5-encoder.gguf", encode_large: bool = False):
        self.path = model_path if isinstance(model_path, Path) else Path(model_path)
        # Configure GGUF Writer
        self.gguf_writer = gguf.GGUFWriter(path=None, arch="t5encoder")
        self._model = None
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

    def prepare_tensors(self):
        self.prepare_encoder()

    def prepare_encoder(self):
        if self.model.text_encoder.config.hidden_size != self.model.decoder.config.hidden_size:
            self.add_tensor("t5encoder.down_proj", self.model.enc_to_dec_proj.weight)
            self.add_tensor("t5encoder.down_proj_bias", self.model.enc_to_dec_proj.bias)
        self.add_tensor("t5encoder.token_embd", self.model.text_encoder.encoder.embed_tokens.weight)
        self.add_tensor("t5encoder.enc.final_layer_norm", self.model.text_encoder.encoder.final_layer_norm.weight)
        for i, layer in enumerate(self.model.text_encoder.encoder.block):
            if i == 0:
                self.add_tensor(f"t5encoder.enc.blk.{i}.attn_rel_b", layer.layer[0].SelfAttention.relative_attention_bias.weight)
            # attention
            self.add_tensor(f"t5encoder.enc.blk.{i}.attn_q", layer.layer[0].SelfAttention.q.weight)
            self.add_tensor(f"t5encoder.enc.blk.{i}.attn_k", layer.layer[0].SelfAttention.k.weight)
            self.add_tensor(f"t5encoder.enc.blk.{i}.attn_v", layer.layer[0].SelfAttention.v.weight)
            self.add_tensor(f"t5encoder.enc.blk.{i}.attn_o", layer.layer[0].SelfAttention.o.weight)
            self.add_tensor(f"t5encoder.enc.blk.{i}.attn_norm", layer.layer[0].layer_norm.weight)
            # mlp
            self.add_tensor(f"t5encoder.enc.blk.{i}.ffn_up", layer.layer[1].DenseReluDense.wi_0.weight)
            self.add_tensor(f"t5encoder.enc.blk.{i}.ffn_gate", layer.layer[1].DenseReluDense.wi_1.weight)
            self.add_tensor(f"t5encoder.enc.blk.{i}.ffn_down", layer.layer[1].DenseReluDense.wo.weight)
            self.add_tensor(f"t5encoder.enc.blk.{i}.ffn_norm", layer.layer[1].layer_norm.weight)

    def add_tensor(self, name, tensor):
        data = tensor.data.detach().to(dtype=torch.float32).numpy()
        self.gguf_writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.F32)

    def prepare_metadata(self):
        total_params, shared_params, expert_params, expert_count = self.gguf_writer.get_total_parameter_count()
        self.metadata = gguf.Metadata.load(None, None, "t5-encoder", total_params)

        # Fallback to model directory name if metadata name is still missing
        if self.metadata.name is None:
            self.metadata.name = "t5-encoder";

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
        self.gguf_writer.arch = "t5encoder"
        self.gguf_writer.add_context_length(self.model.text_encoder.config.n_positions)
        self.gguf_writer.add_embedding_length(self.model.text_encoder.config.d_model)
        self.gguf_writer.add_feed_forward_length(self.model.text_encoder.config.d_ff)
        self.gguf_writer.add_block_count(self.model.text_encoder.config.num_layers)
        self.gguf_writer.add_head_count(self.model.text_encoder.config.num_heads)
        self.gguf_writer.add_vocab_size(self.model.text_encoder.config.vocab_size)
        self.gguf_writer.add_file_type(gguf.LlamaFileType.ALL_F32)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.output_size", self.model.decoder.config.hidden_size)

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
        self.gguf_writer.add_bos_token_id(0)
        self.gguf_writer.add_add_bos_token(False)
        self.gguf_writer.add_add_eos_token(True)

    def write(self):
        self.prepare_tensors()
        self.prepare_metadata()
        self.gguf_writer.write_header_to_file(path=self.path)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file(progress=True)
        self.gguf_writer.close()
