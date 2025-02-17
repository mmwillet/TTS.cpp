### Overview

This script runs a series of benchmarks to test the generative throughput of the TTS.cpp implementation. Over 30 sentences, it aggregates the tokens per second for both the generative model and the decoder model, the real time factor (i.e. the generation time per model divided by the time length of the audio output), and the end to end mean generation in in milliseconds.

### Requirements

* perf_batter and the parler library must be built 
* A local GGUF file for parler tts mini

### Usage

In order to get a detailed breakdown the functionality currently available you can call the cli with the `--help` parameter. This will return a breakdown of all parameters:
```commandline
./perf_battery --help

--n-threads (-nt):
    The number of cpu threads to run generation with. Defaults to 10.
--use-metal (-m):
    (OPTIONAL) whether or not to use metal acceleration.
--no-cross-attn (-ca):
    (OPTIONAL) Whether to not include cross attention
--model-path (-mp):
    (REQUIRED) The local path of the gguf model file for Parler TTS mini v1.
```

General usage should follow from these possible parameters. E.G. The following command will save generated speech to the `/tmp/test.wav` file.

```commandline
./perf_battery --model-path /model/path/to/gguf_file.gguf --use-metal
```
the output will look like the following:
```
Mean Stats for arch Parler-TTS:

  Generation Time (ms):             12439.43255
  Generation Real Time Factor (ms): 1.15635

```

### Latest Results

*Please note that the results listed below are for Parler TTS mini*

The currently measured performance breakdown for Parler Mini v1.0 with Q5_0 quantization without cross attention (i.e. the fastest stable generation with the Parler model) and 32bit floating point weights in the audio decoder:

```
Mean Stats:

  Generation Time (ms):             8599.550347
  Decode Time (ms):                 4228.528055
  Generation TPS:                   1134.453434
  Decode TPS:                       1878.693855
  Generation Real Time Factor (ms): 0.695635
  Decode Real Time Factor (ms):     0.416398
```

Please note that while memory overhead improved a small amount, no substantial difference in inference speed was observed when the audio decoder model was converted from 32bit to 16bit floats.
