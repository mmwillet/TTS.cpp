### Overview

This directory contains a python tool for training phonemization rules and compiling dictionary resources via other phonemization libraries such that they can be run natively from TTS.cpp. This tool uses a completely naive approach which compiles grapheme to phoneme rules based on the rate at which they correspond.

### Phonemization

It is quite common for test-to-speech models, especially smaller models, to use phonemization (i.e. the conversion of words to a standardized phonetically consistent format) in order to reduce a model's need to learn complicated edge cases. While there are several approaches to phonemization, the most common format is the International Phonetic Alphabet (IPA), and the most rapid approaches use rules based algorithms (e.g. [espeak-ng](https://github.com/espeak-ng/) supports a commonly used rules based approach for phonemization). While there are C-based libraries that support phonemization, making them a requirement for TTS.cpp reduces its flexibility and sustainability. As such, TTS.cpp will support both the integration of other phonemizer libs (i.e. [espeak-ng](https://github.com/espeak-ng/)) as well as its own trainable rules based phonemizer with supported gguf encoding. The latter is designed to allow TTS.cpp to effectively word out-of-the-box, but is unlikely to be as flexible or performant as dedicated libraries.

### How it Works and Trains

The TTS.cpp phonemizer works by breaking words up into common grapheme chunks (i.e. groups of letters that can be converted to phonemes) and assigning phonemes to each grapheme depending on the last and next grapheme in the word. Any exceptions that cannot be handled by these rules are stored in a separate dictionary or stored as * match exception (i.e. an grapheme specific exception for words with a similar structure). 

The script present in this directory compiles these rules by using a preexisting phonemizer (in this case espeak) and deterministically matching its behavior by iterating over the [NLTK](https://www.nltk.org/) english word corpus and espeaks generated phonemes per word.

### Usage

It is not recommended that you run this training process natively as it is both time and compute heavy. Instead you can currently download a [precompiled gguf file from huggingface](https://huggingface.co/mmwillet2/TTS_ipa_en_us_phonemizer/tree/main).

#### Installation

If you wish to run this training process locally you must python3.10+ installed locally and a recent version at or greater than 1.52.0 of espeak-ng must be installed (installation instruction can be found [here](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md)).

When the above requirements have been met run the following in this local directory:

```commandline
$ pip3 install -r requirements.txt
```

#### Running Training

Upon completing the installation, training can be run via running the ./train_phonemizer script in this local directory like so:

```commandline
./train_phonemizer --export-path "the/path/to/save.gguf" --checkpoint-directory "/directory/for/checkpoints" --espeak-dylib-path "/the/path/to/your/espeak/dylib/or/dll" --persist
```

If installed via brew the espeak dylib will be found under '/opt/homebrew/Cellar/espeak-ng/*your installed version*/lib/'
