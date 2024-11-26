### Overview

This script runs a series of benchmarks to test the generative throughput of the TTS.cpp implementation. Over 30 sentences, it aggregates the tokens per second for both the generative model and the decoder model, the generation time per model divided by the time length of the audio output, and the end to end mean generation in in milliseconds.

### Requirements

* perf_batter and the parler library must be built 
* A local GGUF file for parler tts mini

### Usage

In order to get a detailed breakdown the functionality currently available you can call the cli with the `--help` parameter. This will return a breakdown of all parameters:
```commandline
./perf_battery --help

--n-threads (-nt):
    The number of cpu threads to run generation with. Defaults to 10.
--model-path (-mp):
    (REQUIRED) The local path of the gguf model file for Parler TTS mini v1.
--use-metal (-m):
    Whether to use metal acceleration.
```

General usage should follow from these possible parameters. E.G. The following command will save generated speech to the `/tmp/test.wav` file.

```commandline
./perf_batter --model-path /model/path/to/gguf_file.gguf --use-metal
```
the output will look like the following:
```
Mean Stats:

  Generation Time (ms):      2707.983408
  Decode Time (ms):          3336.636650
  Generation TPS:            692.753165
  Decode TPS:                562.703661
  Generation by output (ms): 1.119094
  Decode by output (ms):     1.379688
```
