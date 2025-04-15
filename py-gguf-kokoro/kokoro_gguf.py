import gguf
from typing import List, Optional
from pathlib import Path
from kokoro import KPipeline
from huggingface_hub import hf_hub_download
import torch
import json


ALBERT_PARTS = {
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

ALBERT_LAYER_PART = "encoder.albert_layer_groups.0.albert_layers.0."
ALBERT_TOKEN_TYPE_EMB = "embeddings.token_type_embeddings.weight"

DURATION_PREDICTOR_PARTS = {
    'F0_proj.weight': "f0_proj_kernel",
    'F0_proj.bias': "f0_proj_bias",
    'N_proj.weight': "n_proj_kernel",
    'N_proj.bias': "n_proj_bias",
    'duration_proj.linear_layer.weight': "duration_proj",
    'duration_proj.linear_layer.bias': "duration_proj_bias"
}

LSTM_WEIGHTS = ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0', 'weight_ih_l0_reverse', 'weight_hh_l0_reverse', 'bias_ih_l0_reverse', 'bias_hh_l0_reverse']
TTS_PHONEMIZER = 0
ESPEAK = 1
IPA = 0

TTS_PHONEMIZATIION_KEYS = [
    "phonemizer.graphemes",
    "phonemizer.rules.keys",
    "phonemizer.rules.phonemes",
    "phonemizer.dictionary.keys",
    "phonemizer.dictionary.values",
]

VOICES = ['af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_heart', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa', 'am_santa', 'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis']


class KokoroEncoder:
    def __init__(self, model_path: Path | str = "./kokoro.gguf", repo_id: Path | str = 'hexgrad/Kokoro-82M', voices: Optional[List[str]] = None, 
        use_espeak: bool = False, phonemizer_repo: str = "mmwillet2/TTS_ipa_en_us_phonemizer"):
        self.path = model_path if isinstance(model_path, Path) else Path(model_path)
        # Configure GGUF Writer
        self.gguf_writer = gguf.GGUFWriter(path=None, arch="kokoro")
        self._model = None
        self._config = None
        self.repo_id = repo_id
        self.voices = voices or VOICES
        self.use_espeak = use_espeak
        self.phonemizer_repo = phonemizer_repo

    @property
    def model(self):
        if self._model is None:
            self._model = KPipeline(lang_code='a', repo_id=self.repo_id).model
        return self._model

    @property
    def config(self):
        if self._config is None:
            conf_path = hf_hub_download(repo_id=self.repo_id, filename='config.json')
            with open(conf_path, "r+") as f:
                self._config = json.load(f)
        return self._config

    def prepare_tensors(self):
        self.prepare_albert()
        self.prepare_duration_predictor()
        self.prepare_text_encoder()
        self.prepare_decoder()
        self.prepare_voices()

    def prepare_voices(self):
        self.gguf_writer.add_array("kokoro.voices", self.voices)
        vtensors = []
        for voice in self.voices:
            f = hf_hub_download(repo_id=self.repo_id, filename=f'voices/{voice}.pt')
            pack = torch.load(f, weights_only=True)
            vtensors.append(pack.squeeze(1))
        data = torch.stack(tuple(vtensors)).numpy()
        self.gguf_writer.add_tensor(f"kokoro.voice_tensors.{voice}", data, raw_dtype=gguf.GGMLQuantizationType.F32)

    def get_regularized_weight(self, modules, name):
        mname = ".".join(name.split(".")[:-1])
        mod = modules[mname]
        hook = list(mod._forward_pre_hooks.values())[0]
        hook(mod, None)
        return mod.weight

    def prepare_generator_res_block(self, base_name: str, tensor_name: str, param):
        parts = tensor_name.split(".")
        index = parts[1]
        data = param.data.to(dtype=torch.float32).detach().numpy()
        if parts[0][:-1] == "adain":
            if parts[2] == "norm":
                return
            bname = f"beta{parts[0][-1]}"
            gname = f"gamma{parts[0][-1]}"
            data = [data[:data.shape[0]//2],  data[data.shape[0]//2:]]
            self.gguf_writer.add_tensor(f"{base_name}.{index}.{gname}_{parts[-1]}", data[0], raw_dtype=gguf.GGMLQuantizationType.F32)          
            self.gguf_writer.add_tensor(f"{base_name}.{index}.{bname}_{parts[-1]}", data[1], raw_dtype=gguf.GGMLQuantizationType.F32)          
        else:
            nn = f"{base_name}.{index}.{parts[0]}" if parts[-1] not in ["weight", "bias"] else f"{base_name}.{index}.{parts[0]}_{parts[-1]}"
            self.gguf_writer.add_tensor(nn, data, raw_dtype=gguf.GGMLQuantizationType.F32)          

    def prepare_generator(self, base_name: str, tensor_name: str, param):
        parts = tensor_name.split(".")
        data = param.data.to(dtype=torch.float32).detach().numpy()
        if parts[0] == "m_source":
            self.gguf_writer.add_tensor(f"{base_name}.{'_'.join([parts[0], parts[-1]])}", data, raw_dtype=gguf.GGMLQuantizationType.F32)
        elif parts[0] in ["noise_convs", "noise_res"]:
            if parts[0] == "noise_res":
                self.prepare_generator_res_block(f"{base_name}.noise_blocks.{parts[1]}.resblock", ".".join(parts[2:]), param)
            else:
                self.gguf_writer.add_tensor(f"{base_name}.noise_blocks.{parts[1]}.conv_{parts[-1]}", data, raw_dtype=gguf.GGMLQuantizationType.F32)
        elif parts[0] == "ups":
            self.gguf_writer.add_tensor(f"{base_name}.{tensor_name}", data, raw_dtype=gguf.GGMLQuantizationType.F32)
        elif parts[0] == "resblocks":
            self.prepare_generator_res_block(f"{base_name}.{parts[0]}.{parts[1]}", ".".join(parts[2:]), param)
        elif parts[0] == "conv_post":
            self.gguf_writer.add_tensor(f"{base_name}.{'_'.join(parts)}", data, raw_dtype=gguf.GGMLQuantizationType.F32)

    def prepare_decoder(self):
        base = "kokoro.decoder"
        modules = {n: mod for n, mod in self.model.decoder.named_modules()}
        for name, param in self.model.decoder.named_parameters():
            parts = name.split(".")
            if parts[-1] == "weight_v":
                continue
            elif parts[-1] == "weight_g":
                param = self.get_regularized_weight(modules, name)
                parts[-1] = "weight"
            if parts[0] == "generator":
                self.prepare_generator(f"{base}.generator", ".".join(parts[1:]), param)
            elif parts[0] == "decode":
                self.prepare_adain_res_block(f"{base}.decoder_blocks.{parts[1]}", ".".join(parts[2:]), param)
            elif parts[0] == "encode":
                self.prepare_adain_res_block(f"{base}.encoder_block", ".".join(parts[1:]), param)
            elif parts[0] == "F0_conv":
                nn = "_".join(parts)
                data = param.data.to(dtype=torch.float32).detach().numpy()
                self.gguf_writer.add_tensor(f"{base}.{nn.lower()}", data, raw_dtype=gguf.GGMLQuantizationType.F32)          
            elif parts[0] == "N_conv":
                nn = "_".join(parts)
                data = param.data.to(dtype=torch.float32).detach().numpy()
                self.gguf_writer.add_tensor(f"{base}.{nn.lower()}", data, raw_dtype=gguf.GGMLQuantizationType.F32)  
            elif parts[0] == "asr_res":
                data = param.data.to(dtype=torch.float32).detach().numpy()
                self.gguf_writer.add_tensor(f"{base}.asr_conv_{parts[-1]}", data, raw_dtype=gguf.GGMLQuantizationType.F32)  

    def prepare_text_encoder(self):
        base = "kokoro.text_encoder"
        modules = {n: mod for n, mod in self.model.text_encoder.named_modules()}
        for name, param in self.model.text_encoder.named_parameters():
            parts = name.split(".")
            if parts[-1] == "weight_v":
                continue
            elif parts[-1] == "weight_g":
                param = self.get_regularized_weight(modules, name)
                parts[-1] = "weight"
            if "embedding" == parts[0]:
                data = param.data.to(dtype=torch.float32).detach().numpy()
                nn = "_".join(parts)
                self.gguf_writer.add_tensor(f"{base}.{nn}", data, raw_dtype=gguf.GGMLQuantizationType.F32)
            elif parts[0] == "lstm":
                self.prepare_lstm(f"{base}.lstm", parts[1], param)
            elif parts[0] == "cnn":
                layer_index = int(parts[1])
                data = param.data.to(dtype=torch.float32).detach().numpy()
                self.gguf_writer.add_tensor(f"{base}.layers.{layer_index}.{parts[-1]}", data, raw_dtype=gguf.GGMLQuantizationType.F32)

    def prepare_albert(self):
        base = "kokoro.albert"
        for name, param in self.model.bert.named_parameters():
            if name in ALBERT_PARTS:
                data = param.data.to(dtype=torch.float32)
                data = data.detach().numpy()
                self.gguf_writer.add_tensor(f"{base}.{ALBERT_PARTS[name]}", data, raw_dtype=gguf.GGMLQuantizationType.F32)
            elif ALBERT_LAYER_PART in name and name[len(ALBERT_LAYER_PART):] in ALBERT_PARTS:
                data = param.data.to(dtype=torch.float32)
                data = data.detach().numpy()
                self.gguf_writer.add_tensor(f"{base}.layer.0.{ALBERT_PARTS[name[len(ALBERT_LAYER_PART):]]}", data, raw_dtype=gguf.GGMLQuantizationType.F32)
            elif name == ALBERT_TOKEN_TYPE_EMB:
                data = param.data.to(dtype=torch.float32)
                data = data.detach().numpy()[0, :]
                self.gguf_writer.add_tensor(f"{base}.token_type_embd", data, raw_dtype=gguf.GGMLQuantizationType.F32)

    def prepare_lstm(self, base_name: str, tensor_name: str, param):
        data = param.data.to(dtype=torch.float32).detach().numpy()
        data = [data[i*(data.shape[0]//4):(i+1)*(data.shape[0]//4), :] if len(data.shape) > 1 else data[i*(data.shape[0]//4):(i+1)*(data.shape[0]//4)] for i in range(4)]
        layer = int(tensor_name.split("_")[2][1:])
        if "weight" in tensor_name:
            for i, d in enumerate(data):
                index = i*2 if "_ih_" in tensor_name else i*2+1
                name_part = "reverse_weights" if "reverse" in tensor_name else "weights"
                self.gguf_writer.add_tensor(f"{base_name}.{layer}.{name_part}.{index}", d, raw_dtype=gguf.GGMLQuantizationType.F32)
        elif "bias" in tensor_name:
            for i, d in enumerate(data):
                index = i*2 if "_ih_" in tensor_name else i*2+1
                name_part = "reverse_biases" if "reverse" in tensor_name else "biases"
                self.gguf_writer.add_tensor(f"{base_name}.{layer}.{name_part}.{index}", d, raw_dtype=gguf.GGMLQuantizationType.F32)

    def prepare_adain_res_block(self, base: str, tensor_name: str, param):
        parts = tensor_name.split(".")
        data = param.data.to(dtype=torch.float32).detach().detach().numpy()
        if parts[0] in ["norm1", "norm2"]:
            if parts[1] == "norm":
                return # related to affine bug with instance norm; these weight variables aren't actually used.
            data = [data[:data.shape[0]//2], data[data.shape[0]//2:]]
            self.gguf_writer.add_tensor(f"{base}.{parts[0]}_gamma_{parts[-1]}", data[0], raw_dtype=gguf.GGMLQuantizationType.F32)
            self.gguf_writer.add_tensor(f"{base}.{parts[0]}_beta_{parts[-1]}", data[1], raw_dtype=gguf.GGMLQuantizationType.F32)
        else:
            data = param.data.to(dtype=torch.float32).detach().numpy()
            nname = "_".join(parts)
            self.gguf_writer.add_tensor(f"{base}.{nname}", data, raw_dtype=gguf.GGMLQuantizationType.F32)

    def prepare_duration_predictor_layer_tensor(self, base_name: str, tensor_name: str, param):
        parts = tensor_name.split(".")
        index = int(parts[1])
        layer_index = index // 2
        if index % 2 == 1:
            data = param.data.to(dtype=torch.float32).detach().detach().numpy()
            data = [data[:data.shape[0]//2],  data[data.shape[0]//2:]]
            self.gguf_writer.add_tensor(f"{base_name}.{index}.gamma_{parts[-1]}", data[0], raw_dtype=gguf.GGMLQuantizationType.F32)
            self.gguf_writer.add_tensor(f"{base_name}.{index}.beta_{parts[-1]}", data[1], raw_dtype=gguf.GGMLQuantizationType.F32)
        else:
            self.prepare_lstm(f"{base_name}.{index}.lstm", parts[-1], param)

    def prepare_duration_predictor(self):
        base = "kokoro.duration_predictor"
        modules = {n: mod for n, mod in self.model.predictor.named_modules()}
        for name, param in self.model.predictor.named_parameters():
            parts = name.split(".")
            if parts[-1] == "weight_v":
                continue
            elif parts[-1] == "weight_g":
                param = self.get_regularized_weight(modules, name)
                parts[-1] = "weight"
            if "text_encoder" in name:
                self.prepare_duration_predictor_layer_tensor(f"{base}.layers", name[13:], param)
            elif "lstm" in name:
                self.prepare_lstm(f"{base}.duration_lstm", name[5:], param)
            elif "shared" in name:
                self.prepare_lstm(f"{base}.shared_lstm", name[7:], param)
            elif ".".join(parts) in DURATION_PREDICTOR_PARTS:
                data = param.data.to(dtype=torch.float32)
                data = data.detach().numpy()
                self.gguf_writer.add_tensor(f"{base}.{DURATION_PREDICTOR_PARTS[name]}", data, raw_dtype=gguf.GGMLQuantizationType.F32)
            elif parts[0] == "N":
                self.prepare_adain_res_block(f"{base}.n_blocks.{parts[1]}", ".".join(parts[2:]), param)
            elif parts[0] == "F0":
                self.prepare_adain_res_block(f"{base}.f0_blocks.{parts[1]}", ".".join(parts[2:]), param)
        encode_weight = self.model.bert_encoder.weight.data.to(dtype=torch.float32).detach().numpy()
        encode_bias = self.model.bert_encoder.bias.data.to(dtype=torch.float32).detach().numpy()
        self.gguf_writer.add_tensor(f"{base}.encode", encode_weight, raw_dtype=gguf.GGMLQuantizationType.F32)
        self.gguf_writer.add_tensor(f"{base}.encode_bias", encode_bias, raw_dtype=gguf.GGMLQuantizationType.F32)

    def prepare_metadata(self):
        total_params, shared_params, expert_params, expert_count = self.gguf_writer.get_total_parameter_count()
        self.metadata = gguf.Metadata.load(None, None, "kokoro", total_params)

        # Fallback to model directory name if metadata name is still missing
        if self.metadata.name is None:
            self.metadata.name = "kokoro"

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

    def encode_tts_phonemizer(self):
        path = hf_hub_download(repo_id=self.phonemizer_repo, filename='tts_en_us_phonemizer.gguf')
        reader = gguf.GGUFReader(path=path)
        for key in TTS_PHONEMIZATIION_KEYS:
            field = reader.get_field(key)
            data = [str(bytes(field.parts[idx]), encoding='utf-8') for idx in field.data]
            self.gguf_writer.add_array(key, data)
        del reader

    def set_gguf_parameters(self):
        # basic stuff
        self.gguf_writer.add_pad_token_id(0)
        self.gguf_writer.add_decoder_start_token_id(0)

        # Albert duration predictor portion
        self.gguf_writer.arch = "kokoro.duration_predictor.albert"

        # these are just hard coded in the kokoro repo
        self.gguf_writer.add_context_length(512)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.layers", 1)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.attn_heads", self.config["plbert"]["num_attention_heads"])
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.hidden_size", self.config["plbert"]["hidden_size"])
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.recurrence", self.config["plbert"]["num_hidden_layers"])

        # general duration prediction config
        self.gguf_writer.arch = "kokoro.duration_predictor"

        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.hidden_size", self.config["hidden_dim"])
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.layers", self.config["n_layer"])
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.f0_n_blocks", len(self.model.predictor.F0))

        # the general text encoder
        self.gguf_writer.add_uint32("kokoro.text_encoder.layers", self.config["n_layer"])

        # config for generation (mostly istfnet)
        self.gguf_writer.arch = "kokoro.decoder.generator"

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

        # phonemizer
        self.gguf_writer.add_uint32("phonemizer.type", ESPEAK if self.use_espeak else TTS_PHONEMIZER)
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

    def write(self):
        self.prepare_tensors()
        self.prepare_metadata()
        self.gguf_writer.write_header_to_file(path=self.path)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file(progress=True)
        self.gguf_writer.close()
