### Overview

This is a simple cli for running TTS.cpp phonemization on a pass text string. For more information on how TTC.cpp phonemization works, how it is trained, and the motivations for this support please see the [phonemization training readme](../../phonemization_training/README.md).

### Requirements

* phonemize and library must be built 
* A local GGUF file containg TTS.cpp phonemization rules

### Usage

In order to get a detailed breakdown the functionality currently available you can call the cli with the `--help` parameter. This will return a breakdown of all parameters:
```commandline
./build/bin/phonemize --help

--phonemizer-path (-mp):
    (REQUIRED) The local path of the gguf phonemiser file for TTS.cpp phonemizer.
--prompt (-p):
    (REQUIRED) The text prompt to phonemize.
```

General usage should follow from these possible parameters. E.G. The following command will return the phonemized IPA text for the prompt.

```commandline
./build/bin/phonemize --phonemizer-path "/path/to/tts_phonemizer.gguf" --prompt "this is a test."
```
