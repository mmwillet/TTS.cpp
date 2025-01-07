### Overview

This script runs a simple restful HTTP server which supports an OpenAI like `/v1/audio/speech` path for generating and returning TTS content. It supports basic model parallelism and has simple queuing support. This protocol is not currently built for production level services and should not be used for commercial purposes.

### Configuration

In order to get a detailed breakdown of the functionality currently available you can call the tts-server with the `--help` parameter. This will return a breakdown of all parameters:

```commandline
./build/bin/tts-server --help

--temperature (-t):
    (OPTIONAL) The temperature to use when generating outputs by default. Defaults to 0.9.
--repetition-penalty (-r):
    The by channel repetition penalty to be applied to the sampled output of the model by default. defaults to 1.0.
--topk (-tk):
    (OPTIONAL) When set to an integer value greater than 0, generation uses nucleus sampling over topk nucleaus size by default. Defaults to 50.
--n-threads (-nt):
    The number of cpu threads to run generation with. Defaults to hardware concurrency.
--port (-p):
    (OPTIONAL) The port to use. Defaults to 8080.
--n-http-threads (-ht):
    (OPTIONAL) The number of http threads to use. Defaults to hardware concurrency minus 1.
--timeout (-t):
    (OPTIONAL) The server side timeout on http calls in seconds. Defaults to 300 seconds.
--n-parallelism (-np):
    (OPTIONAL) the number of parallel models to run asynchronously. Deafults to 1.
--use-metal (-m):
    (OPTIONAL) Whether to use metal acceleration
--no-cross-attn (-ca):
    (OPTIONAL) Whether to not include cross attention
--model-path (-mp):
    (REQUIRED) The local path of the gguf model file for Parler TTS mini or large v1.
--text-encoder-path (-tep):
    (OPTIONAL) The local path of the text encoder gguf model for conditional generaiton.
--ssl-file-cert (-sfc):
    (OPTIONAL) The local path to the PEM encoded ssl cert.
--ssl-file-key (-sfk):
    (OPTIONAL) The local path to the PEM encoded ssl private key.
--host (-h):
    (OPTIONAL) the hostname of the server. Defaults to '127.0.0.1'.
```

Important configuration here includes `--n-parallelism` which describes how may models for asynchronous processing and `--model-path` which describes from where to load the model locally.

Simple local usage can be achieved via the following simple command:

```commandline
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

```commandline
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
