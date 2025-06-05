### Overview

This is a simple cli for running TTS.cpp phonemization on a pass text string. For more information on how TTC.cpp phonemization works, how it is trained, and the motivations for this support please see the [phonemization training readme](../../phonemization_training/README.md).

### Requirements

* phonemize and library must be built 
* A local GGUF file containg TTS.cpp phonemization rules

### Usage

In order to get a detailed breakdown the functionality currently available you can call the cli with the `--help` parameter. This will return a breakdown of all parameters:
```console
$ ./phonemize --help
--espeak-voice-id (-eid):
    (OPTIONAL) The eSpeak voice id to use for phonemization. This should only be specified when the correct eSpeak voice cannot be inferred from the Kokoro voice. See MultiLanguage Configuration in the README for more info.
--phonemizer-path (-mp):
    (OPTIONAL) The local path of the gguf phonemiser file for TTS.cpp phonemizer. Omit this to use eSpeak to generate phonemes.
--prompt (-p):
    (REQUIRED) The text prompt to phonemize.
```

General usage should follow from these possible parameters. E.G. The following command will return the phonemized IPA text for the prompt via the TTS.cpp phonemizer.

```bash
./build/bin/phonemize --phonemizer-path "/path/to/tts_phonemizer.gguf" --prompt "this is a test."
```

#### Espeak

To use espeak phonemization you must first install the TTS with espeak linked. Phonemization can then be accomplished via the following:

```bash
./build/bin/phonemize --prompt "this is a test." --use-espeak
```
