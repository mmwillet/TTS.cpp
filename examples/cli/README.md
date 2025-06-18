### Overview

This simple example cli tool can be used to generate speach from a text prompt and save it localy to a _.wav_ audio file.

### Requirements

* CLI and library must be built 
* A local GGUF file for parler tts mini

### Usage

In order to get a detailed breakdown the functionality currently available you can call the cli with the `--help` parameter. This will return a breakdown of all parameters:
```bash
./tts-cli --help

--temperature (-t):
    The temperature to use when generating outputs. Defaults to 1.0.
--repetition-penalty (-r):
    The by channel repetition penalty to be applied the sampled output of the model. defaults to 1.0.
--top-p (tp):
    (OPTIONAL) the sum of probabilities to sample over. Must be a value between 0.0 and 1.0. Defaults to 1.0.
--n-threads (-nt):
    The number of cpu threads to run generation with. Defaults to hardware concurrency. If hardware concurrency cannot be determined then it defaults to 1.
--topk (-tk):
    (OPTIONAL) When set to an integer value greater than 0 generation uses nucleus sampling over topk nucleaus size. Defaults to 50.
--max-tokens (-mt):
    (OPTIONAL) The max audio tokens or token batches to generate where each represents approximates 11 ms of audio. Only applied to Dia generation. If set to zero as is its default then the default max generation size. Warning values under 15 are not supported.
--use-metal (-m):
    (OPTIONAL) Whether to use metal acceleration
--no-cross-attn (-ca):
    (OPTIONAL) Whether to not include cross attention
--vad (-va):
    (OPTIONAL) whether to apply voice inactivity detection (VAD) and strip silence form the end of the output (particularly useful for Parler TSS). By default, no VAD is applied.
--play:
    (OPTIONAL) Whether to play back the audio immediately instead of saving it to file.
--model-path (-mp):
    (REQUIRED) The local path of the gguf model file for Parler TTS mini or large v1, Dia, or Kokoro.
--prompt (-p):
    (REQUIRED) The text prompt for which to generate audio in quotation markers.
--save-path (-sp):
    (OPTIONAL) The path to save the audio output to in a .wav format. Defaults to TTS.cpp.wav
--conditional-prompt (-cp):
    (OPTIONAL) A distinct conditional prompt to use for generating. If none is provided the preencoded prompt is used. '--text-encoder-path' must be set to use conditional generation.
--text-encoder-path (-tep):
    (OPTIONAL) The local path of the text encoder gguf model for conditional generaiton.
--voice (-v):
    (OPTIONAL) The voice to use to generate the audio. This is only used for models with voice packs.
--espeak-voice-id (-eid):
    (OPTIONAL) The espeak voice id to use for phonemization. This should only be specified when the correct espeak voice cannot be inferred from the kokoro voice ( see MultiLanguage Configuration in the README for more info).
```

General usage should follow from these possible parameters. E.G. The following command will save generated speech to the `/tmp/test.wav` file.

```bash
./tts-cli --model-path /model/path/to/gguf_file.gguf --prompt "I am saying some words" --save-path /tmp/test.wav
```

#### Dia and Orpheus Generation Arguments

Currently the default cli arguments are not aligned with Dia's or Orpheus' default sampling settings. Specifically the temperature and topk settings should be changed to  `1.3` and `35` respectively when generating with Dia like so:

```bash
./tts-cli --model-path /model/path/to/Dia.gguf --prompt "[S1] Hi, I am Dia, this is how I talk." --save-path /tmp/test.wav --topk 35 --temperature 1.3
```

and the voice, temperature, and repetition penalty setting should be changed to a valid voice (e.g. `leah`), `0.7`, and `1.1` respectively when generating with Orpheus like so:

```bash
./tts-cli --model-path /model/path/to/Orpheus.gguf --prompt "Hi, I am Orpheus, this is how I talk." --save-path /tmp/test.wav --voice leah --temperature 0.7 --repetition-penalty 1.1
```


#### Conditional Generation

Conditional generation is a Parler TTS specific behavior.

By default the Parler TTS model is saved to the GGUF format with a pre-encoded conditional prompt (i.e. a prompt used to determine how to generate speech), but if the text encoder model, the T5-Encoder model, is avaiable in gguf format (see the [python convertion scripts](../../py-gguf/README.md) for more information on how to prepare the T5-Encoder model) then a new conditional prompt can be used for generation like so:

```bash
./tts-cli --model-path /model/path/to/gguf_file.gguf --prompt "I am saying some words" --save-path /tmp/test.wav --text-encoder-path /model/path/to/t5_encoder_file.gguf --consditional-prompt "deep voice"
```

#### Distinct Voice Support

Kokoro and Orpheus both support voices which can be set via the `--voice` (`-v`) argument. Orpheus supports the following voices:

```
"zoe", "zac","jess", "leo", "mia", "julia", "leah"
```

and Kokoro supports the voices listedin the section below.

#### MultiLanguage Configuration

Kokoro supports multiple langauges with distinct voices, and, by default, the standard voices are encoded in the Kokoro gguf file. Below is a list of the available voices:

```
'af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 'af_kore', 'af_nicole',
'af_nova', 'af_river', 'af_sarah', 'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir',
'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa', 'bf_alice', 'bf_emma',
'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis', 'ef_dora',
'em_alex', 'em_santa', 'ff_siwis', 'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi', 'if_sara',
'im_nicola', 'jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo', 'pf_dora',
'pm_alex', 'pm_santa', 'zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi'
```

Each voice has a language assigned and gender assigned to it where the first letter of the pack represents the language and the second the gender (e.g. `af_alloy` is an American English Female voice; `a` corresponds to American Enlgish and `f` to Female). Below is a list of all currently supported langauges mapped to their respective codes:

```
# 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
# 🇪🇸 'e' => Spanish es
# 🇫🇷 'f' => French fr-fr
# 🇮🇳 'h' => Hindi hi
# 🇮🇹 'i' => Italian it
# 🇯🇵 'j' => Japanese: pip install misaki[ja]
# 🇧🇷 'p' => Brazilian Portuguese pt-br
# 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
```

By default when a voice of a specific language is used, phonemization for that language will be automatically detected. However, when multiple phonetic alphabets exist for a single language the default phonemization language might not be appropriate (e.g. Mandarin latin as english is standard for Mandarin, but Pinyin might be preferred). In such cases it is necessary to specify the specific espeak-ng voice file id via the `--espeak-voice-id` argument. A comprehensive list of viable voice ids for this field can be found under the `file` column via the following espeak command:

```bash
espeak-ng --voices
```

