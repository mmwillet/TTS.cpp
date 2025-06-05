# TODO: should not use this
if (WIN32)
    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

# TTS

add_library(tts
            ../include/tts.h
            ../include/phonemizer.h
            tts.cpp
            tokenizer.cpp
            sampler.cpp
            parler_model.cpp
            dac_model.cpp
            util.cpp
            t5_encoder_model.cpp
            phonemizer.cpp
            tts_model.cpp
            kokoro_model.cpp
            dia_model.cpp
            )

target_include_directories(tts PUBLIC . ../include ../ggml/src/)

if (ESPEAK_INCLUDE_DIRS)
    set_source_files_properties(phonemizer.cpp PROPERTIES COMPILE_FLAGS "${ESPEAK_CFLAGS_OTHER}")
    set_source_files_properties(phonemizer.cpp PROPERTIES INCLUDE_DIRECTORIES "${ESPEAK_INCLUDE_DIRS}")
    target_link_libraries(tts PUBLIC ${ESPEAK_LIBRARIES})
endif()
target_link_libraries(tts PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(tts PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(tts PRIVATE LLAMA_BUILD)
    target_compile_definitions(tts PUBLIC  LLAMA_SHARED)
endif()
