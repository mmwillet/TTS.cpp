# TODO: should not use this
if (WIN32)
    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

# parler

add_library(parler
            ../include/parler.h
            ../include/args.h
            ../include/phonemizer.h
            parler.cpp
            tokenizer.cpp
            sampler.cpp
            dac_runner.cpp
            parler_gguf.cpp
            parler_model.cpp
            dac_model.cpp
            util.cpp
            args.cpp
            t5_encoder_model.cpp
            phonemizer.cpp
            )

target_include_directories(parler PUBLIC . ../include ../ggml/src/)
target_compile_features   (parler PUBLIC cxx_std_11) # don't bump

target_link_libraries(parler PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(parler PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(parler PRIVATE LLAMA_BUILD)
    target_compile_definitions(parler PUBLIC  LLAMA_SHARED)
endif()
