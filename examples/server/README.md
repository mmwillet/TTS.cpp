### Overview

This script runs a simple restful HTTP server which supports an OpenAI like `/v1/audio/speech` path for generating and returning TTS content. It supports basic model parallelism and has simple queuing support. This protocol is not currently built for production level services and should not be used for commercial purposes.

### Configuration

In order to get a detailed breakdown of the functionality currently available you can call the tts-server with the `--help` parameter. This will return a breakdown of all parameters:

```console
$ ./tts-server --help
--default-model (-dm):
    (OPTIONAL) The default model to use when multiple models (a directory with multiple GGUF files) are provided. This can be set by giving the path to the model (./models/Kokoro_no_espeak.gguf), the filename (Kokoro_no_espeak.gguf), or the model ID itself (Kokoro_no_espeak).
--espeak-voice-id (-eid):
    (OPTIONAL) The eSpeak voice id to use for phonemization. This should only be specified when the correct eSpeak voice cannot be inferred from the Kokoro voice. See MultiLanguage Configuration in the README for more info.
--host (-h):
    (OPTIONAL) The hostname of the server. Defaults to 127.0.0.1.
--max-tokens (-mt):
    (OPTIONAL) The max audio tokens or token batches to generate where each represents approximates 11 ms of audio. Only applied to Dia generation. If set to zero as is its default then the default max generation size. Warning values under 15 are not supported.
--model-path (-mp):
    (REQUIRED) The local path of the gguf model(s) to load.
--n-http-threads (-ht):
    (OPTIONAL) The number of http threads to use. Defaults to hardware concurrency minus 1.
--n-parallelism (-np):
    (OPTIONAL) The number of parallel models to run asynchronously. Defaults to 1.
--n-threads (-nt):
    (OPTIONAL) The number of CPU threads to run calculations with. Defaults to known hardware concurrency. If hardware concurrency cannot be determined then it defaults to 1.
--no-cross-attn (-ca):
    (OPTIONAL) Whether to not include cross attention.
--port (-p):
    (OPTIONAL) The port to use. Defaults to 8080.
--repetition-penalty (-r):
    (OPTIONAL) The per-channel repetition penalty to be applied the sampled output of the model.
--temperature (-t):
    (OPTIONAL) The temperature to use when generating outputs.
--text-encoder-path (-tep):
    (OPTIONAL) The local path of the text encoder gguf model for conditional generation.
--timeout (-t):
    (OPTIONAL) The server side timeout on http calls in seconds. Defaults to 300 seconds.
--top-p (-mt):
    (OPTIONAL) The sum of probabilities to sample over. Must be a value between 0.0 and 1.0. Defaults to 1.0.
--topk (-tk):
    (OPTIONAL) When set to an integer value greater than 0 generation uses nucleus sampling over topk nucleus size. Defaults to 50.
--use-metal (-m):
    (OPTIONAL) Whether to use metal acceleration.
--voice (-v):
    (OPTIONAL) The voice to use to generate the audio. This is only used for models with voice packs.
```

Important configuration here includes `--n-parallelism` which describes how may models for asynchronous processing and `--model-path` which describes from where to load the model locally.

Simple local usage can be achieved via the following simple command:

```bash
./build/bin/tts-server --model-path /path/to/model/gguf-file.gguf
```

This will run the server on port `8080` via host `127.0.0.1`.

### Usage

The server currently supports three paths:

* `/health` (which functions as a simple health check)
* `/v1/audio/speech` (which returns speech in an audio file format)
* `/v1/audio/conditional-prompt` (which updates the conditional prompt of the active model)
	* currently this is only supported when model parrallism is set to 1. 

The primary endpoint, `/v1/audio/speech`, can be interacted with like so:

```bash
curl http://127.0.0.1:8080/v1/audio/speech  \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This will be converted to speech.",
    "temperature": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.1,
    "response_format": "wav"
  }' \
  --output ./save-path.wav
``` 

The only required parameter is `input` otherwise generation configuration will be determined by the defaults set on server initialization, and the `response_format` will use `wav`. The `response_format` field currently supports only `wav` and `aiff` audio formats.

### Future Work

Future work will include:
* Support for token authentication and permissioning
* Multiple model support
* Streaming audio, for longform audio generation.
