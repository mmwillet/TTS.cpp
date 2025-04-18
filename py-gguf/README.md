### Overview

This directory contains the torch to GGUF format conversion scripts for the [Parler TTS Mini Model](https://huggingface.co/parler-tts/parler-tts-mini-v1), the [Parler TTS Large Model](https://huggingface.co/parler-tts/parler-tts-large-v1), and the [Kokoro-82M TTS model](https://huggingface.co/hexgrad/Kokoro-82M).

These scripts are strictly intended to encode the tensors of the aforementioned models into a 32bit floating point precision format. Quantization and lower precision conversions are performed against compiled models via the [quantization cli](../examples/quantize/README.md).

### Requirements

In order to run the installation and conversion script you will need python3 (tested with 3.9 - 3.11) and [pip3](https://packaging.python.org/en/latest/tutorials/installing-packages/) installed locally.

### Installation

all requisite requirements can be installed via pip like so:
```commandline
pip3 install -r requirements.txt 
```

### Parler TTS

The GGUF conversion script for Parler TTS can be run via the `convert_parler_tts_to_gguf` file locally like so: 
```commandline
python3 ./convert_parler_tts_to_gguf --save-path ./parler-tts-large.gguf --voice-prompt "female voice" --large-model
```

the command accepts _--save-path_ which described where to save the GGUF model file to, the flag _--large-model_ which when passed encodes [Parler-TTS-large](https://huggingface.co/parler-tts/parler-tts-large-v1) (rather than [mini](https://huggingface.co/parler-tts/parler-tts-mini-v1)), _--voice-prompt_ which is a sentence or statement that desribes how the model's voice should sound at generation time, and _--repo-id-override_ which override the huggingface repository to pull the model tensors from (this setting overrides the _--large-model_ argument). 

#### Voice Prompt

The Parler TTS model is trained to alter how it generates audio tokens via cross attending against a text prompt generated via `google/flan-t5-large` a T5-encoder model. In order to avoid this encoding step on the ggml side, this converter generates the prompt's associated hidden states ahead of time and encodes them directly into the gguf model file.

#### Conditional Voice Prompt

If you would like to alter the voice prompt used to generate with parler TTS on the fly you will need to prepare the text encoder model, a T5-encoder model, in the gguf format. This can be accomplished by running `convert_t5_encoder_to_gguf` from this directory:

```commandline
python3 ./convert_t5_encoder_to_gguf --save-path ./t5-encoder-large.gguf --large-model
```

To use this model alongside the parler tts model see the [cli readme for information on conditional generation](../examples/cli/README.md).

### Kokoro

The GGUF conversion script for Kokorocan be run via the `convert_kokoro_to_gguf` file locally like so: 
```commandline
python3 ./convert_kokoro_to_gguf --save-path ./kokoro.gguf
```

the command accepts _--save-path_ which described where to save the GGUF model file to, _--tts-phonemizer_ which when passed encodes the model to use TTS.cpp native phonemization (currently not recommended), and _--repo-id_ which describes the hugging face repo from which to download the model (defaults to 'hexgrad/Kokoro-82M'). Currently all standard Kokoro voices packs are encoded alongside the model (this is not currently customizable through the CLI).
