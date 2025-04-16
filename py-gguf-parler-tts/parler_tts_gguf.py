import gguf
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from parler_tts import ParlerTTSForConditionalGeneration
import torch
from enum import IntEnum
import json
import logging
from typing import Optional, Dict, Union

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Constants ---
# Default Hugging Face model identifiers
DEFAULT_MODEL_ID_MINI = "parler-tts/parler-tts-mini-v1"
DEFAULT_MODEL_ID_LARGE = "parler-tts/parler-tts-large-v1"

# GGUF Constants
GGUF_ARCH = "parler-tts"
GGUF_DECODER_ARCH = f"{GGUF_ARCH}.decoder"
GGUF_DAC_ARCH = f"{GGUF_ARCH}.dac"
GGUF_AUDIO_ARCH = f"{GGUF_ARCH}.audio"

# Mapping from string quant type to GGUF/Torch types
QUANT_TYPE_MAP = {
    "f32": (gguf.GGMLQuantizationType.F32, torch.float32, gguf.LlamaFileType.ALL_F32),
    "f16": (gguf.GGMLQuantizationType.F16, torch.float16, gguf.LlamaFileType.MOSTLY_F16),
    # Add more complex quantizations here (e.g., Q8_0) if needed,
    # which would require implementing the quantization logic itself.
}
SUPPORTED_QUANT_TYPES = list(QUANT_TYPE_MAP.keys())


class SentencePieceTokenTypes(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6

# --- DAC Weight Renaming Maps ---
# These dictionaries map the Hugging Face Diffusers DAC weight names
# to a more structured naming convention suitable for GGUF.

# For 'ResidualUnit' blocks within DAC decoder blocks
DAC_RESIDUAL_UNIT_PARTS: Dict[str, str] = {
    "block.0.alpha": "res.initial.alpha",
    "block.1.bias": "res.initial.bias",
    "block.1.weight": "res.initial.weight",
    "block.2.alpha": "res.final.alpha",
    "block.3.bias": "res.final.bias",
    "block.3.weight": "res.final.weight",
}

# For the main layers within the DAC decoder
DAC_DECODER_PARTS: Dict[str, str] = {
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

# For layers within each DAC 'DecoderBlock'
DAC_DECODER_BLOCK_PARTS: Dict[str, str] = {
    "block.2": "residual_unit.0",
    "block.3": "residual_unit.1",
    "block.4": "residual_unit.2",
    "block.0.alpha": "final.alpha", # Corresponds to the Conv1d after residual units
    "block.1.bias": "final.bias",   # Corresponds to the Conv1d after residual units
    "block.1.weight": "final.weight", # Corresponds to the Conv1d after residual units
}


class ParlerTTSEncoder:
    """
    Encodes Parler TTS models from Hugging Face Hub into GGUF format.

    Handles weight renaming (especially for the DAC components) and metadata setup.
    Supports F32 and F16 quantization types for output.
    """
    def __init__(self,
                 source_model_id: str,
                 output_path: Union[str, Path],
                 output_quant_type: str = "f16"):
        """
        Initializes the encoder.

        Args:
            source_model_id: The Hugging Face model identifier (e.g., "parler-tts/parler-tts-mini-v1").
            output_path: The desired path for the output GGUF file. The filename might be adjusted
                         based on quantization type.
            output_quant_type: The desired quantization type ('f32' or 'f16'). Defaults to 'f16'.
        """
        if output_quant_type.lower() not in SUPPORTED_QUANT_TYPES:
            raise ValueError(f"Unsupported quantization type '{output_quant_type}'. "
                             f"Choose from: {SUPPORTED_QUANT_TYPES}")

        self.source_model_id = source_model_id
        self.output_path_template = Path(output_path)
        self.output_quant_type = output_quant_type.lower()

        # Get target GGUF/Torch types from map
        self.gguf_quant_type, self.torch_dtype, self.gguf_file_type = QUANT_TYPE_MAP[self.output_quant_type]
        self.output_path: Optional[Path] = None # Final path determined in prepare_metadata

        # Configure GGUF Writer - path set later
        self.gguf_writer = gguf.GGUFWriter(path=None, arch=GGUF_ARCH)

        self._model: Optional[ParlerTTSForConditionalGeneration] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self.hparams: Optional[Dict] = None
        self.audio_hparams: Optional[Dict] = None
        self.metadata: Optional[gguf.Metadata] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Target quantization: {self.output_quant_type.upper()}")


    @property
    def model(self) -> ParlerTTSForConditionalGeneration:
        """Lazy loads the Parler TTS model."""
        if self._model is None:
            logger.info(f"Loading model: {self.source_model_id}")
            try:
                self._model = ParlerTTSForConditionalGeneration.from_pretrained(
                    self.source_model_id
                ).to(self.device)
                self._model.eval() # Set to evaluation mode
            except Exception as e:
                logger.exception(f"Failed to load model '{self.source_model_id}'.")
                raise RuntimeError(f"Model loading failed: {e}") from e
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Lazy loads the tokenizer."""
        if self._tokenizer is None:
            logger.info(f"Loading tokenizer for: {self.source_model_id}")
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.source_model_id)
            except Exception as e:
                logger.exception(f"Failed to load tokenizer for '{self.source_model_id}'.")
                raise RuntimeError(f"Tokenizer loading failed: {e}") from e
        return self._tokenizer

    def _add_tensor(self, name: str, tensor: torch.Tensor):
        """
        Adds a tensor to the GGUF writer, converting it to the target dtype.
        Handles potential squeezing and NumPy conversion.
        """
        # Detach tensor from graph, move to CPU (GGUF requires numpy), convert dtype
        data = tensor.detach().cpu().to(dtype=self.torch_dtype)

        # GGUF prefers contiguous tensors
        data = data.contiguous()

        # Convert to NumPy array for GGUF writer
        numpy_data = data.numpy()

        self.gguf_writer.add_tensor(name, numpy_data, raw_dtype=self.gguf_quant_type)
        logger.debug(f"Added tensor '{name}' with shape {numpy_data.shape} and type {self.gguf_quant_type.name}")


    def prepare_tensors(self, text_encoding_prompt: str):
        """Prepares and adds all necessary model tensors to the GGUF writer."""
        logger.info("Preparing and adding model tensors...")
        with torch.no_grad(): # Ensure no gradients are computed
            self.prepare_text_encoding(text_encoding_prompt)
            self.prepare_audio_encoder_tensors()
            self.prepare_decoder_tensors()
        logger.info("Finished preparing tensors.")

    def prepare_text_encoding(self, prompt: str):
        """
        Generates the text encoding for the given prompt and adds it as a tensor.
        This pre-computes the prompt encoding, which is common in some GGUF implementations.
        """
        logger.info(f"Generating text encoding for prompt: '{prompt}'")
        if not prompt:
             logger.warning("Empty text_encoding_prompt provided. The resulting GGUF file might require a prompt at runtime.")
             # Optionally, create a dummy tensor or handle this case as needed by the target inference engine.
             # For now, we'll skip adding the tensor if the prompt is empty.
             # You might need to adjust the loading code in your inference engine accordingly.
             # Add a placeholder metadata field?
             self.gguf_writer.add_bool("parler-tts.decoder.prompt_encoding_missing", True)
             return

        # --- Text Encoding Generation (Adapted from model._prepare_text_encoder_kwargs_for_generation) ---
        # This part uses internal methods of the transformers model. It might break if
        # the underlying library changes significantly.
        try:
            logger.debug("Tokenizing prompt...")
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            model_kwargs = {"input_ids": input_ids}

            logger.debug("Preparing model inputs...")
            # Note: _prepare_model_inputs can modify model_kwargs in place
            inputs_tensor, model_input_name, model_kwargs = self.model._prepare_model_inputs(
                None, self.model.generation_config.bos_token_id, model_kwargs
            )

            logger.debug("Preparing special tokens...")
            # Adjusts bos/eos tokens if needed
            self.model._prepare_special_tokens(self.model.generation_config, is_encoder_decoder=False, device=inputs_tensor.device)

            model_kwargs["use_cache"] = self.model.generation_config.use_cache

            logger.debug("Preparing attention mask...")
            # Create attention mask if not provided
            model_kwargs["attention_mask"] = self.model._prepare_attention_mask_for_generation(
                inputs_tensor, self.model.generation_config.pad_token_id, self.model.generation_config.eos_token_id
            )
            # Pad token ID handling seems different across versions, let's try getting it directly
            pad_token_id = self.model.generation_config.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.pad_token_id # Fallback
            if pad_token_id is None:
                 logger.warning("pad_token_id not found in generation_config or tokenizer.")


            logger.debug("Generating encoder outputs...")
            # This is the core step where text encoding happens
            model_kwargs = self.model._prepare_text_encoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name # Removed generation_config, seems inferred in newer versions
            )

            # --- Extract and Add Tensor ---
            if "encoder_outputs" not in model_kwargs or model_kwargs["encoder_outputs"] is None:
                 raise RuntimeError("Failed to generate 'encoder_outputs' in model_kwargs.")

            encoder_hidden_state = model_kwargs["encoder_outputs"].last_hidden_state

            # Squeeze if batch dimension is 1 (typical for single prompt)
            if encoder_hidden_state.shape[0] == 1:
                encoder_hidden_state = encoder_hidden_state.squeeze(0)

            tensor_name = f"{GGUF_DECODER_ARCH}.text_encoding"
            self._add_tensor(tensor_name, encoder_hidden_state)
            # Add the length of the encoding as metadata
            self.gguf_writer.add_uint32(f"{GGUF_DECODER_ARCH}.encode_length", encoder_hidden_state.shape[0])
            self.gguf_writer.add_bool("parler-tts.decoder.prompt_encoding_missing", False)

        except Exception as e:
            logger.exception("Failed during text encoding preparation.")
            raise RuntimeError(f"Text encoding failed: {e}") from e


    def _prep_dac_weights(self, component: torch.nn.Module, prefix: str):
        """
        Helper function to process weights within DAC components (Decoder, Quantizer).
        Handles weight normalization (weight_g, weight_v) commonly used in GANs/VAEs.
        """
        logger.debug(f"Processing DAC component: {prefix}")
        # Get all modules within the component for hook access
        modules = {name: module for name, module in component.named_modules()}

        for name, param in component.named_parameters():
            original_name = name # Keep original name for module lookup
            current_param = param

            # --- Handle Weight Normalization ---
            # Diffusers often uses weight normalization (weight_g, weight_v)
            # We need to reconstruct the actual 'weight' before saving.
            parts_for_normalized_weights = name.split(".")
            if parts_for_normalized_weights[-1] == "weight_g":
                # Found the scaling part of weight normalization
                module_name = ".".join(parts_for_normalized_weights[:-1])
                target_module = modules.get(module_name)
                if target_module is None or not hasattr(target_module, "_forward_pre_hooks"):
                    logger.warning(f"Could not find module or hooks for {module_name} to handle weight_g.")
                    continue # Skip if module structure is unexpected

                # Trigger the forward pre-hooks (usually torch.nn.utils.weight_norm)
                # to compute the actual 'weight' attribute on the module.
                for hook in target_module._forward_pre_hooks.values():
                    hook(target_module, None) # The second arg (input) is often unused by weight_norm hook

                # Get the newly computed 'weight' attribute from the module
                if not hasattr(target_module, "weight"):
                     logger.warning(f"Module {module_name} lacks 'weight' attribute after hook execution.")
                     continue
                current_param = target_module.weight
                # Modify the name to represent the reconstructed weight
                parts_for_normalized_weights[-1] = "weight"
                name = ".".join(parts_for_normalized_weights)
                logger.debug(f"Reconstructed normalized weight for: {name}")

            elif parts_for_normalized_weights[-1] == "weight_v":
                # Ignore the direction part ('weight_v'), as it's combined
                # with 'weight_g' when reconstructing the full 'weight'.
                logger.debug(f"Skipping normalized weight part: {original_name}")
                continue
            # --- End Weight Normalization Handling ---

            # --- Skip Unneeded Quantizer Weights ---
            if "quantizer" in prefix and "in_proj" in name:
                 # The input projection in the quantizer seems related to encoding audio,
                 # which might not be needed for GGUF-based TTS decoding/synthesis.
                 logger.debug(f"Skipping quantizer input projection: {prefix}.{name}")
                 continue
            # --- End Skip ---

            # --- Rename DAC Decoder Weights ---
            # Apply the structured renaming for DAC decoder layers if applicable
            new_name_parts = [prefix]
            if "decoder" in prefix: # Only apply DAC renaming logic to the decoder part
                # Split the name by '.block' to handle nested structure
                name_parts = name.split(".block")
                try:
                    for i, part in enumerate(name_parts):
                        part_key = f"block{part}" if i > 0 else part
                        if i == 0: # Top-level part within the DAC decoder
                            if part_key not in DAC_DECODER_PARTS:
                                raise KeyError(f"Part '{part_key}' not in DAC_DECODER_PARTS (from name '{name}')")
                            new_name_parts.append(DAC_DECODER_PARTS[part_key])
                        elif i == 1: # Inside a DecoderBlock
                             if part_key not in DAC_DECODER_BLOCK_PARTS:
                                 raise KeyError(f"Part '{part_key}' not in DAC_DECODER_BLOCK_PARTS (from name '{name}')")
                             new_name_parts.append(DAC_DECODER_BLOCK_PARTS[part_key])
                        elif i == 2: # Inside a ResidualUnit within a DecoderBlock
                             if part_key not in DAC_RESIDUAL_UNIT_PARTS:
                                 raise KeyError(f"Part '{part_key}' not in DAC_RESIDUAL_UNIT_PARTS (from name '{name}')")
                             new_name_parts.append(DAC_RESIDUAL_UNIT_PARTS[part_key])
                        else:
                             # This case shouldn't happen with current known DAC structures
                             raise ValueError(f"Unexpected nesting level in DAC name: {name} (parts: {name_parts})")
                except (KeyError, ValueError) as e:
                     logger.error(f"Error renaming DAC weight '{prefix}.{name}': {e}")
                     logger.warning("Using original name as fallback.")
                     # Fallback to original name structure if renaming fails
                     new_name_parts = [prefix, name]
                     # Continue processing other weights
            else:
                 # For non-decoder components (like quantizer), just append the original name
                 new_name_parts.append(name)

            final_name = ".".join(new_name_parts)
            # --- End Renaming ---

            # Add the potentially renamed and reconstructed tensor
            self._add_tensor(final_name, current_param.data)


    def prepare_audio_encoder_tensors(self):
        """Prepares and adds tensors from the audio encoder (DAC decoder and quantizer)."""
        logger.info("Preparing audio encoder (DAC) tensors...")
        # Process DAC Decoder part
        self._prep_dac_weights(self.model.audio_encoder.model.decoder, f"{GGUF_ARCH}.audio_encoder.decoder")
        # Process DAC Quantizer part
        self._prep_dac_weights(self.model.audio_encoder.model.quantizer, f"{GGUF_ARCH}.audio_encoder.quantizer")
        logger.info("Finished preparing audio encoder tensors.")


    def prepare_decoder_tensors(self):
        """Prepares and adds tensors from the main Parler TTS decoder."""
        logger.info("Preparing main decoder tensors...")

        # --- Prompt Embeddings ---
        prompt_emb = self.model.embed_prompts.weight.data
        self._add_tensor(f"{GGUF_DECODER_ARCH}.embed_prompts", prompt_emb)

        # --- Positional Embeddings ---
        # Accessing positional embeddings might differ slightly based on the exact decoder architecture (e.g., BART vs. T5)
        # Assuming a common structure like 'model.decoder.embed_positions.weight'
        try:
             # Common path for BART-like decoders used in ParlerTTS
            positional_embed = self.model.decoder.model.decoder.embed_positions.weight.data
            self._add_tensor(f"{GGUF_DECODER_ARCH}.positional_embed", positional_embed)
        except AttributeError:
             logger.warning("Could not find positional embeddings at standard path 'model.decoder.model.decoder.embed_positions.weight'. Check model architecture.")
             # Add alternative paths here if needed for other decoder types

        # --- Decoder Layer Parameters ---
        decoder_component = self.model.decoder.model.decoder # Navigate to the core decoder layers
        for name, param in decoder_component.named_parameters():
             # Skip positional embeddings if they were handled above
             if "embed_positions" in name:
                 continue
             # Add other decoder parameters
             new_name = f"{GGUF_DECODER_ARCH}.{name}"
             self._add_tensor(new_name, param.data)

        # --- LM Heads Parameters ---
        # Parler uses multiple LM heads, one for each codebook
        for i, lm_head in enumerate(self.model.decoder.lm_heads):
             for name, param in lm_head.named_parameters():
                 # Structure the name like 'decoder.lm_head.0.weight', 'decoder.lm_head.1.bias', etc.
                 new_name = f"{GGUF_DECODER_ARCH}.lm_head.{i}.{name}"
                 self._add_tensor(new_name, param.data)

        logger.info("Finished preparing main decoder tensors.")

    def prepare_metadata(self):
        """Prepares and adds model metadata, vocabulary, and hyperparameters."""
        logger.info("Preparing GGUF metadata...")

        # --- Basic Model Info ---
        total_params, shared_params, expert_params, expert_count = self.gguf_writer.get_total_parameter_count()
        logger.info(f"Total parameters: {total_params:,}")
        if shared_params: logger.info(f"Shared parameters: {shared_params:,}")
        if expert_params: logger.info(f"Expert parameters: {expert_params:,} (Count: {expert_count})")

        # Attempt to load metadata, fallback to model ID
        self.metadata = gguf.Metadata.load(None, None, self.source_model_id, total_params)
        if self.metadata.name is None:
            self.metadata.name = self.source_model_id.split('/')[-1] # Use repo name as fallback

        # Generate parameter size label (e.g., 7B, 13B)
        if self.metadata.size_label is None and total_params > 0:
            self.metadata.size_label = gguf.size_label(total_params, shared_params, expert_params, expert_count)
            logger.info(f"Determined model size label: {self.metadata.size_label}")

        # --- Set Final Output Path ---
        # Use the quantization type name (e.g., "f16") in the filename
        output_type_str = self.output_quant_type.upper() # e.g., F16
        # Ensure parent directory exists
        self.output_path_template.parent.mkdir(parents=True, exist_ok=True)
        # Fill template (if path name contains {ftype}) or append type
        self.output_path = Path(gguf.fill_templated_filename(str(self.output_path_template), output_type_str))
        logger.info(f"Final output path: {self.output_path}")


        # --- Add GGUF Metadata Fields ---
        self.metadata.set_gguf_meta_model(self.gguf_writer) # Adds default fields like arch, param count
        self.gguf_writer.add_architecture(GGUF_ARCH)
        self.gguf_writer.add_uint32(gguf.Keys.Base.PARAM_COUNT, total_params)
        if self.metadata.name: self.gguf_writer.add_name(self.metadata.name)
        if self.metadata.size_label: self.gguf_writer.add_description(f"{self.metadata.name} ({self.metadata.size_label})") # Example description
        self.gguf_writer.add_file_type(self.gguf_file_type)
        self.gguf_writer.add_quantization_version(gguf.GGML_QUANT_VERSION) # Use latest quant version


        # --- Add Model-Specific Parameters ---
        self.set_gguf_parameters()

        # --- Add Vocabulary Info ---
        self.set_vocab()

        logger.info("Finished preparing metadata.")


    def set_gguf_parameters(self):
        """Sets model configuration and hyperparameters in GGUF metadata."""
        logger.debug("Setting GGUF parameters...")
        config = self.model.config
        decoder_config = config.decoder
        audio_encoder_config = config.audio_encoder
        generation_config = self.model.generation_config

        # Store configs for potential later use
        self.hparams = decoder_config.to_dict()
        self.audio_hparams = audio_encoder_config.to_dict()

        # --- General Parameters ---
        pad_token_id = generation_config.pad_token_id if generation_config.pad_token_id is not None else self.tokenizer.pad_token_id
        if pad_token_id is not None:
             self.gguf_writer.add_pad_token_id(pad_token_id)
        else:
             logger.warning("Pad token ID not found.")

        if generation_config.decoder_start_token_id is not None:
             self.gguf_writer.add_decoder_start_token_id(generation_config.decoder_start_token_id)
        elif decoder_config.decoder_start_token_id is not None:
              self.gguf_writer.add_decoder_start_token_id(decoder_config.decoder_start_token_id)
        else:
             logger.warning("Decoder start token ID not found.")


        # --- DAC (Audio Encoder) Parameters ---
        # The DAC structure in ParlerTTS seems relatively fixed.
        self.gguf_writer.add_uint32(f"{GGUF_DAC_ARCH}.up_scaling_factor", 512) # Common DAC upscaling
        # Extract strides and padding from the loaded model config if possible
        try:
             # Accessing strides/padding requires inspecting the actual loaded module layers
             # This assumes the structure seen in common DAC implementations used with Parler
             for i in range(4): # DAC typically has 4 upsampling layers
                 # Layer indices in HF model: model.audio_encoder.model.decoder.model[1] to model[4]
                 layer = self.model.audio_encoder.model.decoder.model[i + 1]
                 # The ConvTranspose1d is usually at layer.block[1]
                 stride = layer.block[1].stride[0]
                 padding = layer.block[1].padding[0]
                 output_padding = layer.block[1].output_padding[0] # Also capture output padding
                 self.gguf_writer.add_uint32(f"{GGUF_DAC_ARCH}.layer_{i}.stride", stride)
                 self.gguf_writer.add_uint32(f"{GGUF_DAC_ARCH}.layer_{i}.padding", padding)
                 self.gguf_writer.add_uint32(f"{GGUF_DAC_ARCH}.layer_{i}.output_padding", output_padding)
                 logger.debug(f"DAC Layer {i}: Stride={stride}, Padding={padding}, OutputPadding={output_padding}")
        except (AttributeError, IndexError) as e:
            logger.warning(f"Could not extract DAC strides/padding dynamically: {e}. Check model structure.")
            # Add fallbacks or raise error if these are critical


        # --- Audio Token Parameters (relative to decoder's view) ---
        if decoder_config.bos_token_id is not None:
             self.gguf_writer.add_uint32(f"{GGUF_AUDIO_ARCH}.bos_token_id", decoder_config.bos_token_id)
        if decoder_config.eos_token_id is not None:
             self.gguf_writer.add_uint32(f"{GGUF_AUDIO_ARCH}.eos_token_id", decoder_config.eos_token_id)


        # --- Decoder Parameters ---
        self.gguf_writer.add_uint32(f"{GGUF_DECODER_ARCH}.hidden_size", self.hparams["hidden_size"])
        self.gguf_writer.add_uint32(f"{GGUF_DECODER_ARCH}.num_output_heads", self.hparams["num_codebooks"]) # Num LM heads = num codebooks
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"]) # Max sequence length for decoder
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_layer_count(self.hparams["num_hidden_layers"]) # Use standard layer count key
        # Vocab size specific to the decoder's output (audio tokens)
        self.gguf_writer.add_uint32(f"{GGUF_DECODER_ARCH}.audio_vocab_size", self.hparams["vocab_size"]) # Decoder vocab size is audio token range + specials
        self.gguf_writer.add_uint32(f"{GGUF_DECODER_ARCH}.codebook_size", self.audio_hparams["codebook_size"]) # Size of each codebook in audio encoder

        # Generation related parameters
        if generation_config.max_length is not None:
            self.gguf_writer.add_uint32(f"{GGUF_DECODER_ARCH}.max_generation_length", generation_config.max_length)

        logger.debug("Finished setting GGUF parameters.")


    def set_vocab(self):
        """Extracts and adds vocabulary and token information to GGUF metadata."""
        logger.info("Setting GGUF vocabulary...")
        tokenizer = self.tokenizer
        try:
             # --- Token List ---
             # Create reverse map: ID -> Token string
             vocab_map = {v: k for k, v in tokenizer.get_vocab().items()}
             # Create list ordered by token ID
             ordered_vocab = [vocab_map[i] for i in range(tokenizer.vocab_size)]
             # Handle SentencePiece prefix space ' ' -> ' '
             ordered_vocab = [token.replace(' ', ' ') for token in ordered_vocab]
             self.gguf_writer.add_token_list(ordered_vocab)
             logger.debug(f"Added {len(ordered_vocab)} tokens to list.")

             # --- Token Scores (Logits) ---
             # This relies on the internal representation of SentencePiece models in transformers
             # May be fragile if transformers changes internal structure
             if hasattr(tokenizer, '_tokenizer') and hasattr(tokenizer._tokenizer, 'model') and hasattr(tokenizer._tokenizer.model,'get_vocab'):
                 raw_vocab_with_scores = tokenizer._tokenizer.model.get_vocab() # Returns list of (token, score)
                 scores_map = dict(raw_vocab_with_scores)
                 scores = [scores_map.get(vocab_map[i], 0.0) for i in range(tokenizer.vocab_size)] # Use 0.0 score if token not found
                 self.gguf_writer.add_token_scores(scores)
                 logger.debug(f"Added {len(scores)} token scores.")
             elif hasattr(tokenizer, '_tokenizer') and hasattr(tokenizer._tokenizer, 'to_str'):
                 # Fallback to parsing JSON string (less reliable)
                 logger.warning("Falling back to parsing tokenizer JSON string for scores. This might be fragile.")
                 tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
                 scores_by_token = {token: score for (token, score) in tokenizer_json['model']['vocab']}
                 scores = [scores_by_token.get(vocab_map[i], 0.0) for i in range(tokenizer.vocab_size)]
                 self.gguf_writer.add_token_scores(scores)
                 logger.debug(f"Added {len(scores)} token scores (from JSON).")
             else:
                 logger.warning("Could not extract token scores. Skipping.")

             # --- Token Types ---
             # Attempt to determine token types (Normal, Control, Unknown, etc.)
             # This also depends on internal SentencePiece details
             token_types = []
             if hasattr(tokenizer, '_tokenizer') and hasattr(tokenizer._tokenizer, 'model') and hasattr(tokenizer._tokenizer.model,'vocab_size'):
                 for i in range(tokenizer.vocab_size):
                      # Heuristic: Check common patterns for SentencePiece
                     if tokenizer.is_special(i): # Checks added tokens, unk, bos, eos etc. -> CONTROL likely?
                          tt = SentencePieceTokenTypes.CONTROL.value
                     elif i < 3: # Often PAD=0, EOS=1, UNK=2 are CONTROL/UNUSED depending on tokenizer
                          if i == tokenizer.unk_token_id: tt = SentencePieceTokenTypes.UNKNOWN.value
                          elif i == tokenizer.eos_token_id or i == tokenizer.bos_token_id: tt = SentencePieceTokenTypes.CONTROL.value
                          elif i == tokenizer.pad_token_id: tt = SentencePieceTokenTypes.UNUSED.value # Often unused in generation
                          else: tt = SentencePieceTokenTypes.CONTROL.value # Default special to control
                     elif vocab_map[i].startswith("<") and vocab_map[i].endswith(">"): # User defined special tokens?
                          tt = SentencePieceTokenTypes.USER_DEFINED.value # Or CONTROL? Heuristic needed.
                     elif len(vocab_map[i]) == 1: # Single byte tokens? Check if it's a byte representation
                          # This heuristic is weak, SentencePiece might map single chars normally.
                          # A better check might involve tokenizer.convert_tokens_to_string([vocab_map[i]])
                          tt = SentencePieceTokenTypes.NORMAL.value # Assume normal unless proven byte
                     else:
                          tt = SentencePieceTokenTypes.NORMAL.value # Default to normal
                     token_types.append(tt)
                 self.gguf_writer.add_token_types(token_types)
                 logger.debug(f"Added {len(token_types)} token types (heuristic).")
             else:
                 logger.warning("Could not extract token types. Skipping.")


             # --- Special Token IDs ---
             # Prefer dynamic lookup over hardcoding
             if tokenizer.bos_token_id is not None: self.gguf_writer.add_bos_token_id(tokenizer.bos_token_id)
             if tokenizer.eos_token_id is not None: self.gguf_writer.add_eos_token_id(tokenizer.eos_token_id)
             if tokenizer.unk_token_id is not None: self.gguf_writer.add_unk_token_id(tokenizer.unk_token_id)
             if tokenizer.sep_token_id is not None: self.gguf_writer.add_sep_token_id(tokenizer.sep_token_id)
             if tokenizer.pad_token_id is not None: self.gguf_writer.add_pad_token_id(tokenizer.pad_token_id) # Re-add pad if available from tokenizer


             # --- Add BOS/EOS Flags ---
             # These flags indicate whether to *automatically* add BOS/EOS during tokenization
             # Check tokenizer configuration or common defaults for the model type
             # For many sequence-to-sequence models, add_bos might be false, add_eos true.
             # Defaulting based on common practice, adjust if needed for Parler specifically.
             add_bos = getattr(tokenizer, 'add_bos_token', False) # Check attribute if exists
             add_eos = getattr(tokenizer, 'add_eos_token', True) # Check attribute if exists
             self.gguf_writer.add_add_bos_token(add_bos)
             self.gguf_writer.add_add_eos_token(add_eos)
             logger.debug(f"Setting add_bos_token={add_bos}, add_eos_token={add_eos}")

        except Exception as e:
            logger.exception("Failed during vocabulary preparation.")
            # Don't raise here, allow writing file even if vocab fails partially? Or raise?
            # raise RuntimeError(f"Vocabulary preparation failed: {e}") from e


    def write(self, text_encoding_prompt: str):
        """
        Performs the complete conversion process: prepares tensors, metadata,
        and writes the final GGUF file.

        Args:
            text_encoding_prompt: The text prompt to pre-encode into the GGUF file.
                                  Can be empty, but the resulting file might require
                                  a prompt at runtime.
        """
        logger.info(f"Starting GGUF conversion for model: {self.source_model_id}")
        logger.info(f"Pre-encoding prompt: '{text_encoding_prompt}'")
        logger.info(f"Target quantization: {self.output_quant_type.upper()}")

        try:
            # --- Preparation ---
            # Ensure model and tokenizer are loaded before accessing configs etc.
            _ = self.model
            _ = self.tokenizer
            self.prepare_tensors(text_encoding_prompt)
            self.prepare_metadata() # Determines final output path

            if self.output_path is None:
                raise RuntimeError("Output path was not set during metadata preparation.")

            # --- Writing ---
            logger.info(f"Writing GGUF header to: {self.output_path}")
            self.gguf_writer.write_header_to_file(path=self.output_path)

            logger.info("Writing GGUF metadata (KV pairs)...")
            self.gguf_writer.write_kv_data_to_file()

            logger.info("Writing GGUF tensors...")
            self.gguf_writer.write_tensors_to_file(progress=True) # Enable progress bar

            logger.info(f"Successfully wrote GGUF file: {self.output_path}")

        except Exception as e:
            logger.exception(f"GGUF conversion failed: {e}")
            # Clean up partially written file if it exists
            if self.output_path and self.output_path.exists():
                 try:
                     self.output_path.unlink()
                     logger.info(f"Removed partially written file: {self.output_path}")
                 except OSError as unlink_e:
                     logger.error(f"Failed to remove partial file {self.output_path}: {unlink_e}")
            raise # Re-raise the exception after cleanup attempt

        finally:
            # --- Cleanup ---
            logger.debug("Closing GGUF writer.")
            self.gguf_writer.close()

            logger.debug("Releasing model and tokenizer objects.")
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            if self.device == torch.device("cuda"):
                logger.debug("Clearing CUDA cache.")
                torch.cuda.empty_cache()

            logger.info("Conversion process finished.")


# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---
    # Choose the model ID
    # SOURCE_MODEL = DEFAULT_MODEL_ID_MINI
    SOURCE_MODEL = DEFAULT_MODEL_ID_LARGE

    # Define the output path (filename might be adjusted based on quant type)
    # Example: "./models/parler-tts-large-v1_{ftype}.gguf" -> "./models/parler-tts-large-v1_F16.gguf"
    repo_name = SOURCE_MODEL.split('/')[-1]
    OUTPUT_FILENAME_TEMPLATE = f"./{repo_name}_{{ftype}}.gguf" # Use {ftype} template

    # Choose quantization type ('f32' or 'f16')
    QUANT_TYPE = "f16"

    # Define the prompt to pre-encode (can be empty)
    PROMPT_TO_ENCODE = "This is an example prompt to pre-encode for the Parler TTS model."
    # PROMPT_TO_ENCODE = "" # Example of empty prompt

    # --- Run Encoder ---
    try:
        encoder = ParlerTTSEncoder(
            source_model_id=SOURCE_MODEL,
            output_path=OUTPUT_FILENAME_TEMPLATE,
            output_quant_type=QUANT_TYPE
        )
        encoder.write(text_encoding_prompt=PROMPT_TO_ENCODE)
        print(f"\nConversion successful!")
        if encoder.output_path:
             print(f"GGUF file saved to: {encoder.output_path.resolve()}")

    except (RuntimeError, ValueError, FileNotFoundError) as main_e:
        print(f"\nConversion failed: {main_e}")
    except Exception as unexpected_e:
        print(f"\nAn unexpected error occurred: {unexpected_e}")
        logger.exception("Unexpected error during conversion.")
