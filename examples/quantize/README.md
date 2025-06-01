### Overview

This script converts a 32bit floating point TTS.cpp GGUF model file to a quantized format. [Quantization](https://huggingface.co/docs/optimum/en/concept_guides/quantization) is a technique used to miniturize weight and bias values in order to reduce overhead memory requirements and reduce inference time. Typically the extent of quantization results in a proportionate though minor impact to model proficiency.

**WARNING** Quantization to smaller modes than Q4_0 is not currently supported and importance matrices are currently not supported.
 
### Requirements

* quantize and the parler library must be built 
* A local GGUF file for parler tts mini

### Usage

**Please note** Quantization and lower precision conversion is currently only supported for Parler TTS models. 

In order to get a detailed breakdown of the functionality currently available you can call the cli with the `--help` parameter. This will return a breakdown of all parameters:
```bash
./quantize --help

--quantized-type (-qt):
    (OPTIONAL) The ggml enum of the quantized type to convert compatible model tensors to. For more information see readme. Defaults to Q4_0 quantization (2).
--n-threads (-nt):
    (OPTIONAL) The number of cpu threads to run the quantization process with. Defaults to known hardware concurrency.
--convert-dac-to-f16 (-df):
    (OPTIONAL) Whether to convert the DAC audio decoder model to a 16 bit float.
--quantize-output-heads (-qh):
    (OPTIONAL) Whether to quantize the output heads. Defaults to false and is true when passed (does not accept a parameter).
--quantize-text-embedding (-qe):
    (OPTIONAL) Whether to quantize the input text embededings (only applicable for Parler TTS). Defaults to false and is true when passed (does not accept a parameter).
--quantize-cross-attn-kv (-qkv):
    (OPTIONAL) Whether to quantize the cross attention keys and values (only applicable for Parler TTS). Defaults to false and is true when passed (does not accept a parameter).
--convert-non-quantized-to-f16 (-nqf):
    (OPTIONAL) Whether or not to convert quantization incompatible tensors to 16 bit precision. Only currently applicable to Kokoror. defaults to false.
--model-path (-mp):
    (REQUIRED) The local path of the gguf model file for Parler TTS mini v1 to quantize.
--quantized-model-path (-qp):
    (REQUIRED) The path to save the model in a quantized format.
```

General usage should follow from these possible parameters. E.G. The following command will save a quantized version of the model using Q4_0 quantization to `/model/path/to/new/gguf_file_q.gguf`:

```bash
./quantize --model-path /model/path/to/gguf_file.gguf --quantized-model-path /model/path/to/new/gguf_file_q.gguf --quantized-type 2 
```
Valid types passed to `--quantized-type` are described by the `ggml_type` enum in GGML:

```cpp
        GGML_TYPE_F16     = 1,
        GGML_TYPE_Q4_0    = 2,
        GGML_TYPE_Q4_1    = 3,
        // GGML_TYPE_Q4_2 = 4, support has been removed
        // GGML_TYPE_Q4_3 = 5, support has been removed
        GGML_TYPE_Q5_0    = 6,
        GGML_TYPE_Q5_1    = 7,
        GGML_TYPE_Q8_0    = 8,
        GGML_TYPE_Q8_1    = 9,
        GGML_TYPE_Q2_K    = 10,
        GGML_TYPE_Q3_K    = 11,
        GGML_TYPE_Q4_K    = 12,
        GGML_TYPE_Q5_K    = 13,
        GGML_TYPE_Q6_K    = 14,
        GGML_TYPE_Q8_K    = 15,
        GGML_TYPE_IQ2_XXS = 16,
        GGML_TYPE_IQ2_XS  = 17,
        GGML_TYPE_IQ3_XXS = 18,
        GGML_TYPE_IQ1_S   = 19,
        GGML_TYPE_IQ4_NL  = 20,
        GGML_TYPE_IQ3_S   = 21,
        GGML_TYPE_IQ2_S   = 22,
        GGML_TYPE_IQ4_XS  = 23,
        GGML_TYPE_I8      = 24,
        GGML_TYPE_Q4_0_4_4 = 31,
        GGML_TYPE_Q4_0_4_8 = 32,
        GGML_TYPE_Q4_0_8_8 = 33,
        GGML_TYPE_TQ1_0   = 34,
        GGML_TYPE_TQ2_0   = 35,
```

### Findings

In general results with quantization have thus far been mixed. While the model's generation speed is improved and it does not completely degrade with full Q4 quantization, full quantization and smaller than Q5_0 quantization is not recommended. With Q4 quantization, the model more frequently repeats words, rarely maintains tonal consistency, and sometimes lengthens speech production unnecessarily. Improvement is observed when the text embeddings, persistent cross attention values, and output heads are not quantized and quantization is limited to Q5 and higher.

#### Approaches to Resolve Inconsisency:

The following approaches were experimented with:

- Avoiding quantization of the persistent cross attention keys and values.
  - **Why**: _Since the voice's distinct qualities are largely determined by the cross attention it is possible that quantization of key and value parameters is damaging consistency._
  - **Result**: _Voice consistency improved, but not completely (This could be reflection of the Parler itself being somewhat inconsistent)._
- Cluster sampling 
  - **Why**: _Generally the model performs better with lower temperatures or no sampling, but often fails to terminate on time without sampling. Reducing room for error while still sampling will likely improve quality._
  - **Result**: _Did not improve the model's performance noticeably over simple tempature changes. It is possible that it reduces catastrophic edge cases._
- Adding a static prompt with non-quantized generation (via a persistent cache) before each generation.
  - **Why**: _The model rarely changes its voice mid generation so having an upfront pattern to generate from likely improve consistency._
  - **Result**: _Drastically improved voice consistency_.
  
#### Performance Observations

A clear improvement in tokens per second via the generative model is observed with quantization. Seen below Parler TTS mini with Q5_0 quantization, the model is capable of completing its generation in real time (it generates tokens faster than it takes to listen to them), and the model's TPS has improved from ~693 to ~986.

```
Mean Stats:

  Generation Time (ms):      1945.434146
  Decode Time (ms):          3416.610760
  Generation TPS:            986.040513
  Decode TPS:                562.941718
  Generation by output (ms): 0.786301
  Decode by output (ms):     1.379292
```
