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
    The temperature to use when generating outputs. Defaults to 0.7.
--repetition-penalty (-r):
    The by channel repetition penalty to be applied the sampled output of the model. defaults to 1.1.
--n-threads (-nt):
    The number of cpu threads to run generation with. Defaults to 10.
--topk (-tk):
    (OPTIONAL) when set to an integer value greater than 0 generation uses nucleus sampling over topk nucleaus size
--use-metal (-m):
    (OPTIONAL) Whether to use metal acceleration
--no-cross-attn (-ca):
    (OPTIONAL) Whether to not include cross attention
--model-path (-mp):
    (REQUIRED) The local path of the gguf model file for Parler TTS mini v1.
--prompt (-p):
    (REQUIRED) The text prompt for which to generate audio in quotation markers.
--save-path (-sp):
    (REQUIRED) The path to save the audio output to in a .wav format.
```

General usage should follow from these possible parameters. E.G. The following command will save generated speech to the `/tmp/test.wav` file.

```commandline
./cli --model-path /model/path/to/gguf_file.gguf --prompt "I am saying some words" --save-path /tmp/test.wav
```
