## TTS.cpp

[Roadmap](https://github.com/users/mmwillet/projects/1) / [Modified GGML](https://github.com/mmwillet/ggml/tree/support-for-tts)

### Purpose and Goals

The general purpose of this repository is to support real time generation with open source TTS (_text to speech_) models across common device architectures using the [GGML tensor library](https://github.com/ggerganov/ggml). Rapid STT (_speach to text_), embedding generation, and LLM generation are well supported on GGML (via [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and [llama.cpp](https://github.com/ggerganov/llama.cpp) respectively). As such, this repo seeks to compliment those functionalities with a similarly optimized and portable TTS library.

In this endeavor, MacOS and metal support will be treated as a the primary platform, and, as such, functionality will initially be developed for MacOS and later extended to other OS.   

### Supported Functionality

**Warning!** *Currently TTS.cpp should be treated as a _proof of concept_ and is subject to further development. Existing functionality has not be tested outside of a MacOS X environment.*

#### Model Support

Currently [Parler TTS Mini v0.1](https://huggingface.co/ylacombe/parler_tts_mini_v0.1) is the only supported TTS model.

Additional Model support will initially be added based on open source model performance in the [TTS model arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena).

#### Functionality

| Planned Functionality | OS X       | Linux | Windows |
|----------------------|------------|---|---|
| Basic CPU Generation | &check;    |&cross;|&cross;|
| Metal Acceleration   | &check;_*_ | _ | _ |
| CUDA support         | _          |&cross;|&cross;|
| Quantization         | &check;_*_ |&cross;|&cross;|
| Layer Offloading     | &cross;    |&cross;|&cross;|
| Server Support       | &cross;    |&cross;|&cross;|
| Vulkan Support       | _          |&cross;|&cross;|
| Kompute Support      | _          |&cross;|&cross;|
| Streaming Audio      | &cross;    |&cross;|&cross;|

 _*_ Currently only the generative model supports these.
### Installation

**WARNING!** This library is only currently supported on OS X

#### Requirements:

* Local GGUF format model file (see [py-gguf](./py-gguf/README.md) for information on how to convert the hugging face model to GGUF).
* C++17 and C17
  * XCode Command Line Tools (via `xcode-select --install`) should suffice for OS X
* CMake (>=3.14) 
* GGML pulled locally
  * this can be accomplished via `git clone -b support-for-tts git@github.com:mmwillet/ggml.git`
  
#### Build:

Assuming that the above requirements are met the library and basic CLI example can be built by running the following command in the repository's base directory:
```commandline
cmake -B build                                           
cmake --build build --config Release
```

The CLI executable will be in the `./build/cli` directory and the compiled library will be in the `./build/src` (currently it is named _parler_ as that is the only supported model).

### Usage

See the [CLI example readme](./examples/cli/README.md) for more details on its general usage.
