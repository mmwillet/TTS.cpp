### Overview

This script runs a series of benchmarks to test the generative throughput of the TTS.cpp implementation. Over 30 sentences, it aggregates the tokens per second for both the generative model and the decoder model, the real time factor (i.e. the generation time per model divided by the time length of the audio output), and the end to end mean generation in in milliseconds.

### Requirements

* perf_batter and the parler library must be built 
* A local GGUF file for parler tts mini

### Usage

In order to get a detailed breakdown the functionality currently available you can call the cli with the `--help` parameter. This will return a breakdown of all parameters:
```console
$ ./perf_battery --help
--espeak-voice-id (-eid):
    (OPTIONAL) The eSpeak voice id to use for phonemization. This should only be specified when the correct eSpeak voice cannot be inferred from the Kokoro voice. See MultiLanguage Configuration in the README for more info.
--max-tokens (-mt):
    (OPTIONAL) The max audio tokens or token batches to generate where each represents approximates 11 ms of audio. Only applied to Dia generation. If set to zero as is its default then the default max generation size. Warning values under 15 are not supported.
--model-path (-mp):
    (REQUIRED) The local path of the gguf model(s) to load.
--n-threads (-nt):
    (OPTIONAL) The number of CPU threads to run calculations with. Defaults to known hardware concurrency. If hardware concurrency cannot be determined then it defaults to 1.
--no-cross-attn (-ca):
    (OPTIONAL) Whether to not include cross attention.
--repetition-penalty (-r):
    (OPTIONAL) The per-channel repetition penalty to be applied the sampled output of the model.
--temperature (-t):
    (OPTIONAL) The temperature to use when generating outputs.
--top-p (-mt):
    (OPTIONAL) The sum of probabilities to sample over. Must be a value between 0.0 and 1.0. Defaults to 1.0.
--topk (-tk):
    (OPTIONAL) When set to an integer value greater than 0 generation uses nucleus sampling over topk nucleus size. Defaults to 50.
--use-metal (-m):
    (OPTIONAL) Whether to use metal acceleration.
--voice (-v):
    (OPTIONAL) The voice to use to generate the audio. This is only used for models with voice packs.
```

General usage should follow from these possible parameters. E.G. The following command will save generated speech to the `/tmp/test.wav` file.

```bash
./perf_battery --model-path /model/path/to/gguf_file.gguf --use-metal
```
the output will look like the following:
```text
Mean Stats for arch Parler-TTS:

  Generation Time (ms):             12439.43255
  Generation Real Time Factor (ms): 1.15635

```

### Latest Results

*Please note that the results listed below are for Parler TTS mini*

The currently measured performance breakdown for Parler Mini v1.0 with Q5_0 quantization without cross attention (i.e. the fastest stable generation with the Parler model) and 32bit floating point weights in the audio decoder:

```text
Mean Stats:

  Generation Time (ms):             8599.550347
  Decode Time (ms):                 4228.528055
  Generation TPS:                   1134.453434
  Decode TPS:                       1878.693855
  Generation Real Time Factor (ms): 0.695635
  Decode Real Time Factor (ms):     0.416398
```

Please note that while memory overhead improved a small amount, no substantial difference in inference speed was observed when the audio decoder model was converted from 32bit to 16bit floats.
