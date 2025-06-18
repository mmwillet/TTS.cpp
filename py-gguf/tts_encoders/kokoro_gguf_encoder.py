import gguf
from typing import List, Optional, Dict
from pathlib import Path
from kokoro import KPipeline, KModel
from huggingface_hub import hf_hub_download
from .tts_encoder import TTSEncoder
from .tensor_util import get_regularized_weight
import torch
import torch.nn as nn
import json

# ALBERT_PARTS is a mapping of the torch parameter names in Kokoro's Albert nn.Module to names
# interpretable by TTS.cpp and thereby saved to the GGUF file.
ALBERT_PARTS: Dict[str, str] = {
    "embeddings.word_embeddings.weight": "token_embd",
    "embeddings.position_embeddings.weight": "position_embd",
    "embeddings.LayerNorm.weight": "norm",
    "embeddings.LayerNorm.bias": "norm_bias",
    "encoder.embedding_hidden_mapping_in.weight": "embd",
    "encoder.embedding_hidden_mapping_in.bias": "embd_bias",
    "full_layer_layer_norm.weight": "attn_norm",
    "full_layer_layer_norm.bias": "attn_norm_bias",
    "attention.query.weight": "q",
    "attention.query.bias": "q_bias",
    "attention.key.weight": "k",
    "attention.key.bias": "k_bias",
    "attention.value.weight": "v",
    "attention.value.bias": "v_bias",
    "attention.dense.weight": "o",
    "attention.dense.bias": "o_bias",
    "attention.LayerNorm.weight": "ffn_norm",
    "attention.LayerNorm.bias": "ffn_norm_bias",
    "ffn.weight": "ffn",
    "ffn.bias": "ffn_bias",
    "ffn_output.weight": "ffn_out",
    "ffn_output.bias": "ffn_out_bias"
}

# ALBERT_LAYER_PART is the nn.Module parameter name that demotes Kokoro's Albert layer
ALBERT_LAYER_PART = "encoder.albert_layer_groups.0.albert_layers.0."
# ALBERT_TOKEN_TYPE_EMB is the nn.Module parameter name that demotes Kokoro's Albert Embedding weight
ALBERT_TOKEN_TYPE_EMB = "embeddings.token_type_embeddings.weight"

# DURATION_PREDICTOR_PARTS is a mapping of the nn.Module parameter name of Kokoro's Duration Predictor pytorch
# nn.Module to TTS.cpp interpretable names stored in the GGUF file.
DURATION_PREDICTOR_PARTS: Dict[str, str] = {
    'F0_proj.weight': "f0_proj_kernel",
    'F0_proj.bias': "f0_proj_bias",
    'N_proj.weight': "n_proj_kernel",
    'N_proj.bias': "n_proj_bias",
    'duration_proj.linear_layer.weight': "duration_proj",
    'duration_proj.linear_layer.bias': "duration_proj_bias"
}

# The follow are standard enum values for TTS.cpp phonemization settings used on Kokoro
TTS_PHONEMIZER = 0
ESPEAK_PHONEMIZER = 1
ESPEAK = 2
IPA = 1

# TTS_PHONEMIZATION_KEYS are the keys that the TTS.cpp phonemizer expects and thereby must be transplanted from
# a given phonemizer GGUF file.
TTS_PHONEMIZATION_KEYS: List[str] = [
    "phonemizer.graphemes",
    "phonemizer.rules.keys",
    "phonemizer.rules.phonemes",
    "phonemizer.dictionary.keys",
    "phonemizer.dictionary.values",
]

# Below is a list of the voices to pull by default from the Kokoro repository.
VOICES: List[str] = ['af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 'af_kore', 'af_nicole',
                     'af_nova', 'af_river', 'af_sarah', 'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir',
                     'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa', 'bf_alice', 'bf_emma',
                     'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis', 'ef_dora',
                     'em_alex', 'em_santa', 'ff_siwis', 'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi', 'if_sara',
                     'im_nicola', 'jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo', 'pf_dora',
                     'pm_alex', 'pm_santa', 'zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi']

DEFAULT_KOKORO_REPO = 'hexgrad/Kokoro-82M'
DEFAULT_TTS_PHONEMIZER_REPO = "mmwillet2/TTS_ipa_en_us_phonemizer"
KOKORO_ARCHITECTURE = "kokoro"


class KokoroEncoder(TTSEncoder):
    """
    The purpose of this class is to encode and write the tensors and model configuration for the Kokoro TTS model that
    into a GGUF file.

    General Usage:

    ```python
    from tts_encoders import KokoroEncoder, DEFAULT_KOKORO_REPO

    gguf_encoder = KokoroEncoder("some/local/path.gguf", DEFAULT_KOKORO_REPO, use_espeak=True)
    gguf_encoder.write()
    ```
    """
    def __init__(self, model_path: Path | str = "./kokoro.gguf", repo_id: Path | str = DEFAULT_KOKORO_REPO,
                 voices: Optional[List[str]] = None, use_espeak: bool = False,
                 phonemizer_repo: Path | str = DEFAULT_TTS_PHONEMIZER_REPO):
        """
        :param Path or str model_path: The path to save the generated GGUF file.
        :param Path or str repo_id: The path or repository from which to pull the Kokoro model, its voice embeddings,
            and its configuration.
        :param List[str] voices: the voice names to pull from the repository and include in the generated GGUF file.
        :param bool use_espeak: Denotes whether to assign espeak as the phonemizer (Otherwise defaults to TTS phonemization)
        :param Path or str phonemizer_repo: The path or repository to pull the TTS GGUF encoded phonemizer if not using
            espeak.
        """
        super().__init__(model_path=model_path, architecture=KOKORO_ARCHITECTURE)
        self._model = None
        self._config = None
        self.repo_id = repo_id
        self.voices = voices or VOICES
        self.use_espeak = use_espeak
        self.phonemizer_repo = phonemizer_repo

    @property
    def model(self) -> KModel:
        if self._model is None:
            try:
                # the language code does not impact the model pulled down from the repository.
                self._model = KPipeline(lang_code="a", repo_id=self.repo_id).model
            except Exception as e:
                self.logger.exception(
                    f"Failed with exception, {e}, attempting to load Kokoro via repository '{self.repo_id}'."
                )
                raise e
        return self._model

    @property
    def config(self):
        if self._config is None:
            try:
                conf_path = hf_hub_download(repo_id=self.repo_id, filename='config.json')
            except Exception as e:
                self.logger.exception(
                    f"Failed with exception, {e}, attempting to obtain config.json via repository '{self.repo_id}'."
                )
                raise e
            with open(conf_path, "r+") as f:
                self._config = json.load(f)
        return self._config

    def prepare_tensors(self):
        """
        Implementation of TTSEncoder's abstract method see TTSEncoder for more information
        """
        self.prepare_albert_tensors()
        self.prepare_duration_predictor_tensors()
        self.prepare_text_encoder_tensors()
        self.prepare_decoder_tensors()
        self.prepare_voices()

    def prepare_voices(self):
        """
        Pulls the voice tensors from the Huggingface repository and encodes them as tensors for the GGUF file
        """
        self.gguf_writer.add_array("kokoro.voices", self.voices)
        for voice in self.voices:
            try:
                f = hf_hub_download(repo_id=self.repo_id, filename=f'voices/{voice}.pt')
                pack = torch.load(f, weights_only=True).squeeze(1).numpy()
                self.set_tensor(f"kokoro.voice_tensors.{voice}", pack)
            except Exception as e:
                self.logger.exception(
                    f"Failed with exception, {e}, attempting to obtain voice pack at path 'voices/{voice}.pt' in repository '{self.repo_id}'."
                )
                raise e

    def prepare_generator_res_block_tensor(self, base_name: str, tensor_name: str, param: nn.Parameter):
        """
        Prepares and adds a tensor belonging to an ADA Residual block within the Generator of the Kokoro model
        to the GGUF writer.

        :param str base_name: the name of the residual block
        :param str tensor_name: The sub-name of the tensor in the residual block's context
        :param nn.Parameter param: The torch parameter to encode to the GGUF file
        """
        parts = tensor_name.split(".")
        index = parts[1]
        if parts[0][:-1] == "adain":
            if parts[2] == "norm":
                # affine is often set on the InstanceNorm nn.Modules in Kokoro to bypass a bug in pytorch
                # involving the buffers used by the ADA norms implemented for Kokoro. This allocates an unused
                # parameter; ignore it.
                return
            bname = f"beta{parts[0][-1]}"
            gname = f"gamma{parts[0][-1]}"
            data = param.data.to(dtype=torch.float32).detach().numpy()
            data = [data[:data.shape[0]//2],  data[data.shape[0]//2:]]
            self.set_tensor(f"{base_name}.{index}.{gname}_{parts[-1]}", data[0])
            self.set_tensor(f"{base_name}.{index}.{bname}_{parts[-1]}", data[1])
        else:
            nn = f"{base_name}.{index}.{parts[0]}" if parts[-1] not in ["weight", "bias"] else f"{base_name}.{index}.{parts[0]}_{parts[-1]}"
            self.set_tensor(nn, param)

    def prepare_generator_tensor(self, base_name: str, tensor_name: str, param: nn.Parameter):
        """
        Prepares and adds a tensor belonging to the Generator of the Kokoro model to the GGUF writer.

        :param str base_name: the name of the generator.
        :param str tensor_name: The sub-name of the tensor in the generator's context.
        :param nn.Parameter param: The torch parameter to encode to the GGUF file.
        """
        parts = tensor_name.split(".")
        if parts[0] == "m_source":
            self.set_tensor(f"{base_name}.{'_'.join([parts[0], parts[-1]])}", param)
        elif parts[0] in ["noise_convs", "noise_res"]:
            if parts[0] == "noise_res":
                self.prepare_generator_res_block_tensor(f"{base_name}.noise_blocks.{parts[1]}.resblock", ".".join(parts[2:]), param)
            else:
                self.set_tensor(f"{base_name}.noise_blocks.{parts[1]}.conv_{parts[-1]}", param)
        elif parts[0] == "ups":
            self.set_tensor(f"{base_name}.{tensor_name}", param)
        elif parts[0] == "resblocks":
            self.prepare_generator_res_block_tensor(f"{base_name}.{parts[0]}.{parts[1]}", ".".join(parts[2:]), param)
        elif parts[0] == "conv_post":
            self.set_tensor(f"{base_name}.{'_'.join(parts)}", param)

    def prepare_decoder_tensors(self):
        """
        Prepares the tensors belonging to Kokoro's Decoder nn.Module and adds them to the GGUF writer.
        """
        base = "kokoro.decoder"
        modules = {n: mod for n, mod in self.model.decoder.named_modules()}
        for name, param in self.model.decoder.named_parameters():
            parts = name.split(".")
            if parts[-1] == "weight_v":
                # We will get the normalized weight when we encounter "weight_g" so we should ignore "weight_v"
                continue
            elif parts[-1] == "weight_g":
                param = get_regularized_weight(modules, name)
                parts[-1] = "weight"
            if parts[0] == "generator":
                self.prepare_generator_tensor(f"{base}.generator", ".".join(parts[1:]), param)
            elif parts[0] == "decode":
                self.prepare_adain_res_block_tensor(f"{base}.decoder_blocks.{parts[1]}", ".".join(parts[2:]), param)
            elif parts[0] == "encode":
                self.prepare_adain_res_block_tensor(f"{base}.encoder_block", ".".join(parts[1:]), param)
            elif parts[0] == "F0_conv":
                nn = "_".join(parts)
                self.set_tensor(f"{base}.{nn.lower()}", param)
            elif parts[0] == "N_conv":
                nn = "_".join(parts)
                self.set_tensor(f"{base}.{nn.lower()}", param)
            elif parts[0] == "asr_res":
                self.set_tensor(f"{base}.asr_conv_{parts[-1]}", param)

    def prepare_text_encoder_tensors(self):
        """
        Prepares the tensors belonging to Kokoro's TextEncoder nn.Module and adds them to the GGUF writer.
        """
        base = "kokoro.text_encoder"
        modules = {n: mod for n, mod in self.model.text_encoder.named_modules()}
        for name, param in self.model.text_encoder.named_parameters():
            parts = name.split(".")
            if parts[-1] == "weight_v":
                # We will get the normalized weight when we encounter "weight_g" so we should ignore "weight_v"
                continue
            elif parts[-1] == "weight_g":
                param = get_regularized_weight(modules, name)
                parts[-1] = "weight"
            if "embedding" == parts[0]:
                nn = "_".join(parts)
                self.set_tensor(f"{base}.{nn}", param)
            elif parts[0] == "lstm":
                self.prepare_lstm_tensor(f"{base}.lstm", parts[1], param)
            elif parts[0] == "cnn":
                layer_index = int(parts[1])
                self.set_tensor(f"{base}.layers.{layer_index}.{parts[-1]}", param)

    def prepare_albert_tensors(self):
        """
        Prepares the tensors belonging to Kokoro's ALBERT nn.Module and adds them to the GGUF writer.
        """
        base = "kokoro.albert"
        for name, param in self.model.bert.named_parameters():
            if name in ALBERT_PARTS:
                self.set_tensor(f"{base}.{ALBERT_PARTS[name]}", param)
            elif ALBERT_LAYER_PART in name and name[len(ALBERT_LAYER_PART):] in ALBERT_PARTS:
                self.set_tensor(f"{base}.layer.0.{ALBERT_PARTS[name[len(ALBERT_LAYER_PART):]]}", param)
            elif name == ALBERT_TOKEN_TYPE_EMB:
                data = param.data.to(dtype=torch.float32)
                data = data.detach().numpy()[0, :]
                self.set_tensor(f"{base}.token_type_embd", data)

    def prepare_lstm_tensor(self, base_name: str, tensor_name: str, param):
        """
        Prepares and adds a tensor belonging to an LSTM nn.Module with the Kokoro model to the GGUF writer.

        :param str base_name: the name of the LSTM instance.
        :param str tensor_name: The sub-name of the tensor in the LSTM.
        :param nn.Parameter param: The torch parameter to encode to the GGUF file
        """
        data = param.data.to(dtype=torch.float32).detach().numpy()
        data = [data[i*(data.shape[0]//4):(i+1)*(data.shape[0]//4), :] if len(data.shape) > 1 else data[i*(data.shape[0]//4):(i+1)*(data.shape[0]//4)] for i in range(4)]
        layer = int(tensor_name.split("_")[2][1:])
        if "weight" in tensor_name:
            for i, d in enumerate(data):
                index = i*2 if "_ih_" in tensor_name else i*2+1
                name_part = "reverse_weights" if "reverse" in tensor_name else "weights"
                self.set_tensor(f"{base_name}.{layer}.{name_part}.{index}", d)
        elif "bias" in tensor_name:
            for i, d in enumerate(data):
                index = i*2 if "_ih_" in tensor_name else i*2+1
                name_part = "reverse_biases" if "reverse" in tensor_name else "biases"
                self.set_tensor(f"{base_name}.{layer}.{name_part}.{index}", d)

    def prepare_adain_res_block_tensor(self, base: str, tensor_name: str, param):
        """
        Prepares and adds a tensor belonging to an ADA Residual Block  n.Module with the Kokoro model to the GGUF writer.

        :param str base_name: the name of the residual block
        :param str tensor_name: The sub-name of the tensor within the residual block
        :param nn.Parameter param: The torch parameter to encode to the GGUF file
        """
        parts = tensor_name.split(".")
        if parts[0] in ["norm1", "norm2"]:
            if parts[1] == "norm":
                # This is related to affine bug with instance norm; these weight variables aren't actually used.
                return 
            data = param.data.to(dtype=torch.float32).detach().detach().numpy()
            data = [data[:data.shape[0]//2], data[data.shape[0]//2:]]
            self.set_tensor(f"{base}.{parts[0]}_gamma_{parts[-1]}", data[0])
            self.set_tensor(f"{base}.{parts[0]}_beta_{parts[-1]}", data[1])
        else:
            nname = "_".join(parts)
            self.set_tensor(f"{base}.{nname}", param)

    def prepare_duration_predictor_layer_tensor(self, base_name: str, tensor_name: str, param):
        """
        Prepares and adds a tensor belonging to one of Kokoro Duration Predictor's layers to the GGUF writer.

        :param str base_name: the name of the layer's ModuleList
        :param str tensor_name: The sub-name of the tensor within layer's ModuleList
        :param nn.Parameter param: The torch parameter to encode to the GGUF file
        """
        parts = tensor_name.split(".")
        index = int(parts[1])
        if index % 2 == 1:
            data = param.data.to(dtype=torch.float32).detach().detach().numpy()
            data = [data[:data.shape[0]//2],  data[data.shape[0]//2:]]
            self.set_tensor(f"{base_name}.{index}.gamma_{parts[-1]}", data[0])
            self.set_tensor(f"{base_name}.{index}.beta_{parts[-1]}", data[1])
        else:
            self.prepare_lstm_tensor(f"{base_name}.{index}.lstm", parts[-1], param)

    def prepare_duration_predictor_tensors(self):
        """
        Prepares the tensors belonging to Kokoro's Duration Predictor nn.Module and adds them to the GGUF writer.
        """
        base = "kokoro.duration_predictor"
        modules = {n: mod for n, mod in self.model.predictor.named_modules()}
        for name, param in self.model.predictor.named_parameters():
            parts = name.split(".")
            if parts[-1] == "weight_v":
                # We will get the normalized weight when we encounter "weight_g" so we should ignore "weight_v"
                continue
            elif parts[-1] == "weight_g":
                param = get_regularized_weight(modules, name)
                parts[-1] = "weight"
            if "text_encoder" in name:
                self.prepare_duration_predictor_layer_tensor(f"{base}.layers", name[13:], param)
            elif "lstm" in name:
                self.prepare_lstm_tensor(f"{base}.duration_lstm", name[5:], param)
            elif "shared" in name:
                self.prepare_lstm_tensor(f"{base}.shared_lstm", name[7:], param)
            elif ".".join(parts) in DURATION_PREDICTOR_PARTS:
                self.set_tensor(f"{base}.{DURATION_PREDICTOR_PARTS[name]}", param)
            elif parts[0] == "N":
                self.prepare_adain_res_block_tensor(f"{base}.n_blocks.{parts[1]}", ".".join(parts[2:]), param)
            elif parts[0] == "F0":
                self.prepare_adain_res_block_tensor(f"{base}.f0_blocks.{parts[1]}", ".".join(parts[2:]), param)
        self.set_tensor(f"{base}.encode", self.model.bert_encoder.weight)
        self.set_tensor(f"{base}.encode_bias", self.model.bert_encoder.bias)

    def prepare_metadata(self):
        """
        Implementation of TTSEncoder's Abstract method see TTSEncoder for more information
        """
        total_params, shared_params, expert_params, expert_count = self.gguf_writer.get_total_parameter_count()
        self.metadata = gguf.Metadata.load(None, None, "kokoro", total_params)

        # Generate parameter weight class (useful for leader boards) if not yet determined
        if self.metadata.size_label is None and total_params > 0:
            self.metadata.size_label = gguf.size_label(total_params, shared_params, expert_params, expert_count)

        # Filename Output
        self.set_type()
        self.metadata.set_gguf_meta_model(self.gguf_writer)
        self.set_gguf_parameters()
        self.set_vocab()
        self.gguf_writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    def encode_tts_phonemizer(self):
        """
        Downloads the TTS.cpp phonemizer from HuggingFace and adds its encoded fields to the GGUF writer
        """
        try:
            path = hf_hub_download(repo_id=self.phonemizer_repo, filename='tts_en_us_phonemizer.gguf')
        except Exception as e:
            self.logger.exception(f"Failed to load phonemizer GGUF file, 'tts_en_us_phonemizer.gguf', from repositor, '{self.phonemizer_repo}'")
            raise e
        reader = gguf.GGUFReader(path=path)
        for key in TTS_PHONEMIZATION_KEYS:
            field = reader.get_field(key)
            data = [str(bytes(field.parts[idx]), encoding='utf-8') for idx in field.data]
            self.gguf_writer.add_array(key, data)

    def set_gguf_parameters(self):
        """
        Adds general model configuration to the GGUf writer
        """
        self.gguf_writer.add_pad_token_id(0)
        self.gguf_writer.add_decoder_start_token_id(0)

        # ---- Albert ----
        self.gguf_writer.arch = "kokoro.duration_predictor.albert"

        # These configurations are currently static and hard coded in the kokoro repo
        self.gguf_writer.add_context_length(512)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.layers", 1)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.attn_heads", self.config["plbert"]["num_attention_heads"])
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.hidden_size", self.config["plbert"]["hidden_size"])
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.recurrence", self.config["plbert"]["num_hidden_layers"])

        # ---- Duration Predictor ----
        self.gguf_writer.arch = "kokoro.duration_predictor"

        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.hidden_size", self.config["hidden_dim"])
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.layers", self.config["n_layer"])
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.f0_n_blocks", len(self.model.predictor.F0))

        # ---- Text Encoder ----
        self.gguf_writer.add_uint32(f"{KOKORO_ARCHITECTURE}.text_encoder.layers", self.config["n_layer"])

        # ---- Generator ----
        self.gguf_writer.arch = f"{KOKORO_ARCHITECTURE}.decoder.generator"

        # This is needed to determine the output buffer for the the model in ggml, but isn't needed in torch
        # as a result I am hard coding it here. It can be calculated by determining dividing the output shape by
        #sum of the predicted token durations.
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.up_sampling_factor", 600)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.kernels", self.model.decoder.generator.num_kernels)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.upsamples", self.model.decoder.generator.num_upsamples)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.layers", len(self.model.decoder.decode))
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.padding", self.model.decoder.generator.conv_post.padding[0])
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.n_fft", self.config["istftnet"]["gen_istft_n_fft"])
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.hop", self.config["istftnet"]["gen_istft_hop_size"])

        for i, res in enumerate(self.model.decoder.generator.noise_res):
            for ii in range(3):
                self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.noise_blocks.{i}.res_block.{ii}.padding", res.convs1[ii].padding[0])
                self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.noise_blocks.{i}.res_block.{ii}.dilation", res.convs1[ii].dilation[0])

        for i, conv in enumerate(self.model.decoder.generator.noise_convs):
            self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.noise_blocks.{i}.stride", conv.stride[0])
            self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.noise_blocks.{i}.padding", conv.padding[0])

        for i, res in enumerate(self.model.decoder.generator.resblocks):
            for ii in range(3):
                self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.res_blocks.{i}.{ii}.padding", res.convs1[ii].padding[0])
                self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.res_blocks.{i}.{ii}.dilation", res.convs1[ii].dilation[0])

        for i, up in enumerate(self.model.decoder.generator.ups):
            self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.up_convs.{i}.padding", up.padding[0])
            self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.up_convs.{i}.stride", up.stride[0])

        # ---- Phonemizer ----
        self.gguf_writer.add_uint32("phonemizer.type", ESPEAK_PHONEMIZER if self.use_espeak else TTS_PHONEMIZER)
        self.gguf_writer.add_uint32("phonemizer.phoneme_type", IPA)
        if (not self.use_espeak):
            self.encode_tts_phonemizer()

        self.gguf_writer.arch = "kokoro"
        self.gguf_writer.add_file_type(gguf.LlamaFileType.ALL_F32)

    def set_type(self):
        self.gguf_writer.add_type(gguf.GGUFType.MODEL)

    def set_vocab(self):
        reversed_vocab = {v:k for k, v in self.model.vocab.items()}
        vocab = [""] + [reversed_vocab[i+1] if i+1 in reversed_vocab else "" for i in range(max(reversed_vocab.keys()))]
        self.gguf_writer.add_token_list(vocab)
        self.gguf_writer.add_eos_token_id(0)
