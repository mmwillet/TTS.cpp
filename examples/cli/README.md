### Overview

This simple example cli tool can be used to generate speach from a text prompt and save it localy to a _.wav_ audio file.

### Requirements

* CLI and library must be built 
* A local GGUF file for parler tts mini

### Usage

In order to get a detailed breakdown the functionality currently available you can call the cli with the `--help` parameter. This will return a breakdown of all parameters:
```commandline
./cli --help

--temperature (-t):
    The temperature to use when generating outputs. Defaults to 0.9.
--repetition-penalty (-r):
    The by channel repetition penalty to be applied the sampled output of the model. defaults to 1.1.
--n-threads (-nt):
    The number of cpu threads to run generation with. Defaults to 10.
--topk (-tk):
    (OPTIONAL) when set to an integer value greater than 0 generation uses nucleus sampling over topk nucleaus size. Defaults to 50.
--use-metal (-m):
    (OPTIONAL) Whether to use metal acceleration
--no-cross-attn (-ca):
    (OPTIONAL) Whether to not include cross attention
--model-path (-mp):
    (REQUIRED) The local path of the gguf model file for Parler TTS mini or large v1.
--prompt (-p):
    (REQUIRED) The text prompt for which to generate audio in quotation markers.
--save-path (-sp):
    (REQUIRED) The path to save the audio output to in a .wav format.
--conditional-prompt (-cp):
    (OPTIONAL) A distinct conditional prompt to use for generating. If none is provided the preencoded prompt is used. '--text-encoder-path' must be set to use conditional generation.
--text-encoder-path (-tep):
    (OPTIONAL) The local path of the text encoder gguf model for conditional generaiton.
--voice (-v):
    (OPTIONAL) The voice to use to generate the audio. This is only used for models with voice packs.
```

General usage should follow from these possible parameters. E.G. The following command will save generated speech to the `/tmp/test.wav` file.

```commandline
./cli --model-path /model/path/to/gguf_file.gguf --prompt "I am saying some words" --save-path /tmp/test.wav
```

#### Conditional Generation

By default the Parler TTS model is saved to the GGUF format with a pre-encoded conditional prompt (i.e. a prompt used to determine how to generate speech), but if the text encoder model, the T5-Encoder model, is avaiable in gguf format (see the [python convertion scripts](../../py-gguf/README.md) for more information on how to prepare the T5-Encoder model) then a new conditional prompt can be used for generation like so:

```commandline
./cli --model-path /model/path/to/gguf_file.gguf --prompt "I am saying some words" --save-path /tmp/test.wav --text-encoder-path /model/path/to/t5_encoder_file.gguf --consditional-prompt "deep voice"
```
