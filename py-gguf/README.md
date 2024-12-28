### Overview

This directory contains a simple torch to gguf format conversion script for the [Parler TTS Mini Model](https://huggingface.co/parler-tts/parler-tts-mini-v1) or the [Parler TTS Large Model](https://huggingface.co/parler-tts/parler-tts-large-v1).

Please note that the model encoding pattern used here is extremely naive and subject to further development (especially in order to align its pattern with gguf patterns in llama.cpp ad whisper.cpp).

### Requirements

In order to run the installation and conversion script you will need python3 and [pip3](https://packaging.python.org/en/latest/tutorials/installing-packages/) installed locally.

### Installation

all requisite requirements can be installed via pip:
```commandline
pip3 install -r requirements.txt 
```

### Usage

The gguf conversion script can be run via the `convert_parerl_tts_to_gguf` file locally like so: 
```commandline
python3 ./convert_parler_tts_to_gguf --save-path ./parler-tts-large.gguf --voice-prompt "female voice" --large-model
```

the command accepts _--save-path_ which described where to save the gguf model file to, the flag _--large-model_ which when passed encodes [Parler-TTS-large](https://huggingface.co/parler-tts/parler-tts-large-v1) (rather than [mini](https://huggingface.co/parler-tts/parler-tts-mini-v1)), and _--voice-prompt_ which is a sentence or statement that desribes how the model's voice should sound at generation time.


#### Voice Prompt

The Parler TTS model is trained to alter how it generates audio tokens via cross attend against a text prompt generated via `google/flan-t5-large` a T5-encoder model. In order to avoid this encoding step on the ggml side, this converter generates the prompt's associated hidden states ahead of time and encodes them directly into the gguf model file.
