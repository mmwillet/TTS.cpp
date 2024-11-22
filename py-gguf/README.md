### Overview

This directory contains a simple torch to gguf file conversion script for the Parler TTS Mini Model.

**Please note that the model encoding pattern used here is extremely naive and subject to further development** (especially in order to align its pattern with gguf patterns in llama.cpp ad whisper.cpp).

### Installation

all requisite requirements can be installed via pip:
```commandline
pip3 install -r requirements.txt 
```

### Usage

The gguf conversion script can be run via the `convert_parerl_tts_to_gguf` file locally like so: 
```commandline
python3 ./convert_parler_tts_to_gguf --save-path ./test.gguf
```

the command accepts _--save-path_ which described where to save the gguf model file to and _--voice-prompt_ which is a sentence which is a sentence desribing how the model's voice should sound.

#### Voice Prompt

The Parler TTS model is trained to alter how it generates audio tokens via cross attend against a text prompt generated via `google/flan-t5-large` a T5-encoder model. In order to avoid this encoding step on the ggml side, this converter generates the prompt's associated hidden states ahead of time and encodes them directly into the gguf model file.
