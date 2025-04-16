### Overview

This directory contains a simple torch to gguf format conversion script for the [Kokoro-82M TTS model](https://huggingface.co/hexgrad/Kokoro-82M).

Please note that the model encoding pattern used here is subject to further development (especially in order to align its pattern with gguf patterns in llama.cpp ad whisper.cpp).

### Requirements

In order to run the installation and conversion script you will need python3 and [pip3](https://packaging.python.org/en/latest/tutorials/installing-packages/) installed locally.

### Installation

all requisite requirements can be installed via pip:
```commandline
pip3 install -r requirements.txt 
```

### Usage

The gguf conversion script can be run via the `convert_kokoro_to_gguf` file locally like so: 
```commandline
python3 ./convert_kokoro_to_gguf --save-path ./kokoro.gguf
```

the command accepts _--save-path_ which described where to save the gguf model file to and _--repo-id_ which describes the hugging face repo from which to download the model (defaults to 'hexgrad/Kokoro-82M').
