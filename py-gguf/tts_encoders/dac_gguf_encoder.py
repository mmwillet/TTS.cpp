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


class DACEncoder(TTSEncoder):
    @property
    def dac_model(self):
        raise NotImplementedError("Implmentations of 'DACEncoder' must define #dac_model.")

    def prepare_dac_audio_encoder_tensors(self):
        """
        Prepares and writes the tensors for the DAC audio encoder model used post-generation to conform TTS outputs
        into audio format to the GGUF file writer.
        """
        modules = {name: module for name, module in self.dac_model.decoder.named_modules()}
        for name, param in self.dac_model.decoder.named_parameters():
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

        modules = {name: module for name, module in self.dac_model.quantizer.named_modules()}
        for name, param in self.dac_model.quantizer.named_parameters():
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

    def set_dac_config(self, bos_token_id: int, eos_token_id: int):
        # ---- DAC Audio Encoder configuration ----
        # the upscaling factor represents the input to output upscale rate in the DAC Audio encoder.
        # It is static for all versions of DAC used by Parler-TTS
        self.gguf_writer.add_uint32("dac.up_scaling_factor", 512)
        for i in range(4):
            self.gguf_writer.add_uint32(f"dac.dac_layer_stride_{i}", self.dac_model.decoder.model[i+1].block[1].stride[0])
            self.gguf_writer.add_uint32(f"dac.dac_layer_padding_{i}", self.dac_model.decoder.model[i+1].block[1].padding[0])

        # DAC audio token configuration
        self.gguf_writer.add_uint32(f"audio.bos_token_id", bos_token_id)
        self.gguf_writer.add_uint32(f"audio.eos_token_id", eos_token_id)
