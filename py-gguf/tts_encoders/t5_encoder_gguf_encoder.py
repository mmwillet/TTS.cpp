import gguf
from pathlib import Path
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import json
from .tts_encoder import TTSEncoder
from .parler_tts_gguf_encoder import DEFAULT_PARLER_REPO_MINI_ID


# T5_ARCHITECTURE denotes the model architecture to use in order to interpret the GGUF file written by the T5Encoder below
T5_ARCHITECTURE = "t5encoder"


class T5Encoder(TTSEncoder):
    """
    The purpose of this class is to encode and write the tensors and model configuration for the T5-Encoder that belongs
    to the Parler-TTS model into a GGUF file.

    General Usage:

    ```python
    from t5_encoder_gguf_encoder import T5Encoder, DEFAULT_PARLER_REPO_MINI_ID

    gguf_encoder = T5Encoder("some/local/path.gguf", DEFAULT_PARLER_REPO_MINI_ID)
    gguf_encoder.write()
    ```

    The T5 Encoder model which Parler TTS uses is a modified version of Google's T5-Flan model.
    This protocol is built to mimic the end configuration of a GGUF T5 encoder encoded via llama.cpp, but
    the end model *is* different from a standard T5Encoder model.
    """
    def __init__(self, model_path: Path | str = "./t5-encoder.gguf", repo_id: Path | str = DEFAULT_PARLER_REPO_MINI_ID):
        """
        :param Path or str model_path: the path to save the GGUF file.
        :param Path or str repo_id: the hugging face repo or local path from which to load the pytorch Parler-TTS model.
        """
        super().__init__(model_path=model_path, architecture=T5_ARCHITECTURE)
        self.path = model_path if isinstance(model_path, Path) else Path(model_path)
        # Configure GGUF Writer
        self.repo_id = repo_id
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
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
    def tokenizer(self):
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
        Implementation of TTSEncoder's abstract method see TTSEncoder for more information
        """
        if self.model.text_encoder.config.hidden_size != self.model.decoder.config.hidden_size:
            self.set_tensor("t5encoder.down_proj", self.model.enc_to_dec_proj.weight)
            self.set_tensor("t5encoder.down_proj_bias", self.model.enc_to_dec_proj.bias)
        self.set_tensor("t5encoder.token_embd", self.model.text_encoder.encoder.embed_tokens.weight)
        self.set_tensor("t5encoder.enc.final_layer_norm", self.model.text_encoder.encoder.final_layer_norm.weight)
        for i, layer in enumerate(self.model.text_encoder.encoder.block):
            if i == 0:
                self.set_tensor(f"t5encoder.enc.blk.{i}.attn_rel_b", layer.layer[0].SelfAttention.relative_attention_bias.weight)
            # attention
            self.set_tensor(f"t5encoder.enc.blk.{i}.attn_q", layer.layer[0].SelfAttention.q.weight)
            self.set_tensor(f"t5encoder.enc.blk.{i}.attn_k", layer.layer[0].SelfAttention.k.weight)
            self.set_tensor(f"t5encoder.enc.blk.{i}.attn_v", layer.layer[0].SelfAttention.v.weight)
            self.set_tensor(f"t5encoder.enc.blk.{i}.attn_o", layer.layer[0].SelfAttention.o.weight)
            self.set_tensor(f"t5encoder.enc.blk.{i}.attn_norm", layer.layer[0].layer_norm.weight)
            # mlp
            self.set_tensor(f"t5encoder.enc.blk.{i}.ffn_up", layer.layer[1].DenseReluDense.wi_0.weight)
            self.set_tensor(f"t5encoder.enc.blk.{i}.ffn_gate", layer.layer[1].DenseReluDense.wi_1.weight)
            self.set_tensor(f"t5encoder.enc.blk.{i}.ffn_down", layer.layer[1].DenseReluDense.wo.weight)
            self.set_tensor(f"t5encoder.enc.blk.{i}.ffn_norm", layer.layer[1].layer_norm.weight)

    def prepare_metadata(self):
        """
        Implementation of TTSEncoder's abstract method see TTSEncoder for more information
        """
        total_params, shared_params, expert_params, expert_count = self.gguf_writer.get_total_parameter_count()
        self.metadata = gguf.Metadata.load(None, None, "t5-encoder", total_params)

        # Generate parameter weight class (useful for leader boards) if not yet determined
        if self.metadata.size_label is None and total_params > 0:
            self.metadata.size_label = gguf.size_label(total_params, shared_params, expert_params, expert_count)

        self.set_type()
        self.metadata.set_gguf_meta_model(self.gguf_writer)
        self.set_gguf_parameters()
        self.set_vocab()
        self.gguf_writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    def set_gguf_parameters(self):
        """
        Adds general model configuration to the GGUF writer.
        """
        self.gguf_writer.arch = T5_ARCHITECTURE
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
