## TTS.cpp

[Roadmap](https://github.com/users/mmwillet/projects/1) / [Modified GGML](https://github.com/mmwillet/ggml/tree/support-for-tts)

### Purpose and Goals

The general purpose of this repository is to support real time generation with open source TTS (_text to speech_) models across common device architectures using the [GGML tensor library](https://github.com/ggerganov/ggml). Rapid STT (_speach to text_), embedding generation, and LLM generation are well supported on GGML (via [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and [llama.cpp](https://github.com/ggerganov/llama.cpp) respectively). As such, this repo seeks to compliment those functionalities with a similarly optimized and portable TTS library.

In this endeavor, MacOS and metal support will be treated as the primary platform, and, as such, functionality will initially be developed for MacOS and later extended to other OS.   

### Supported Functionality

**Warning!** *Currently TTS.cpp should be treated as a _proof of concept_ and is subject to further development. Existing functionality has not be tested outside of a MacOS X environment.*

#### Model Support

**Kokoro** is the recommended model. It reliably produces articulate and coherent speech for a variety of prompt sizes. Most of the other models are too large (read: slow), but may support finer-grained voice customization.

| Models | CPU | Metal Acceleration | Quantization | GGUF files |
|--------------------------------------------------------------------------|-------|-------|-------|--------------------------------------------------------|
| [Parler TTS Mini](https://huggingface.co/parler-tts/parler-tts-mini-v1)  |&check;|&check;|&check;|[here](https://huggingface.co/mmwillet2/Parler_TTS_GGUF)|
| [Parler TTS Large](https://huggingface.co/parler-tts/parler-tts-large-v1)|&check;|&check;|&check;|[here](https://huggingface.co/mmwillet2/Parler_TTS_GGUF)|
| [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)                      |&check;|&cross;|&check;|[here](https://huggingface.co/mmwillet2/Kokoro_GGUF)    |
| [Dia](https://github.com/nari-labs/dia)                                  |&check;|&check;|&check;|[here](https://huggingface.co/mmwillet2/Dia_GGUF)       |
| [Orpheus](https://github.com/canopyai/Orpheus-TTS)                       |&check;|&cross;|&cross;|[here](https://huggingface.co/mmwillet2/Orpheus_GGUF)       |

Additional Model support will initially be added based on open source model performance in both the [old TTS model arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) and [new TTS model arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2) as well as the availability of said models' architectures and checkpoints.

#### Functionality

| Planned Functionality | OS X       | Linux | Windows |
|-----------------------|------------|-------|---------|
| Basic CPU Generation  | &check;    |&check;| &cross; |
| Metal Acceleration    | &check;    | _     | _       |
| CUDA support          | _          |&cross;| &cross; |
| Quantization          | &check;_*_ |&cross;| &cross; |
| Layer Offloading      | &cross;    |&cross;| &cross; |
| Server Support        | &check;    |&check;| &cross; |
| Vulkan Support        | _          |&cross;| &cross; |
| Kompute Support       | _          |&cross;| &cross; |
| Streaming Audio       | &cross;    |&cross;| &cross; |

 _*_ Currently only the generative model supports these.
### Installation

**WARNING!** This library is only currently supported on OS X

#### Requirements:

* Local GGUF format model file (see [py-gguf](./py-ggufs/README.md) for information on how to convert the hugging face models to GGUF).
* C++17 and C17
  * XCode Command Line Tools (via `xcode-select --install`) should suffice for OS X
* CMake (>=3.14) 
* GGML pulled locally
  * this can be accomplished via `git clone -b support-for-tts git@github.com:mmwillet/ggml.git`

#### GGML Patch

The local GGML library includes several required patches to the main branch of GGML (making the current TTS ggml branch out of date with modern GGML). Specifically these patches include major modifications to the convolutional transposition operation as well as several new GGML operations which have been implemented for TTS specific purposes; these include `ggml_reciprocal`, `ggml_round`, `ggml_mod`, `ggml_cumsum`, STFT, and iSTFT operations.

We are currently [working on upstreaming some of these operations inorder to deprecate this patch requirement going forward](https://github.com/mmwillet/TTS.cpp/issues/66).

#### Build:

Assuming that the above requirements are met the library and basic CLI example can be built by running the following command in the repository's base directory:
```commandline
cmake -B build                                           
cmake --build build --config Release
```

The CLI executable and other exceutables will be in the `./build` directory (e.g. `./build/cli`) and the compiled library will be in the `./build/src` (currently it is named _parler_ as that is the only supported model).

If you wish to install TTS.cpp with Espeak-ng phonemization support, first [install Espeak-ng](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md). Depending on your installation method the path of the installed library will vary. Upon identifying the installation path to espeak-ng (it should contain `./lib`, `./bin`, `./include`, and `./share` directories), you can compile TTS.cpp with espeak phonemization support by running the follwing in the repositories base directory:

```bash
export ESPEAK_INSTALL_DIR=/absolute/path/to/espeak-ng/dir
cmake -B build
cmake --build build --config Release
```

On Linux, you don't need to manually download or `export` anything. Our build system will automatically detect the development packages installed on your machine:

```bash
# Change `apt` and the package names to match your distro
sudo apt install build-essential cmake # Minimum requirements
sudo apt install git libespeak-ng-dev libsdl2-dev pkg-config # Optional requirements
cmake -B build
cmake --build build --config Release
```

### Usage

See the [CLI example readme](./examples/cli/README.md) for more details on its general usage.

### Quantization and Lower Precision Models

See the [quantization cli readme](./examples/quantize/README.md) for more details on its general usage and behavior. **Please note** Quantization and lower precision conversion is currently only supported for Parler TTS models. 

### Performance

 Given that the central goal of this library is to support real time speech generation on OS X, generation speed has only been rigorously tested in that environment with supported models (i.e. Parler Mini version 1.0).

 With the introduction of metal acceleration support for the DAC audio decoder model, text to speech generation is nearly possible in real time on a standard Apple M1 Max with ~3GB memory overhead. The best real time factor for accelerated models is currently 1.112033. This means that for every second of generated audio, the accelerated models require approximately 1.112033 seconds of generation time (with Q5_0 quantization applied to the generative model). For the latest stats via the performance battery see the [readme therein](./examples/perf_battery/README.md).

# License

Unless indicated otherwise, this repo is `MIT`-licensed.

To the extent required by law, parts derived from the models' original implementations retain their original `Apache-2.0` license. This may include hyperparameters and post-processing logic, but excludes our port to ggml and C++. This makes the resulting binary `Apache-2.0`-licensed if those models are compiled in.

If eSpeak NG support is enabled, the resulting binary is `GPL-3.0-or-later`-licensed.
