# TODO: should not use this
if (WIN32)
    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif ()
endif ()

# TTS

add_library(tts
        ../include/args.h
        tokenizer.cpp
        sampler.cpp
        sampler.h
        util.cpp
        util.h
        args.cpp
        tts_model.cpp
        tts_model.h
)

target_include_directories(tts PUBLIC . ../include ../ggml/src/)

target_link_libraries(tts PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(tts PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(tts PRIVATE LLAMA_BUILD)
    target_compile_definitions(tts PUBLIC LLAMA_SHARED)
endif ()

add_subdirectory(decoder)
add_subdirectory(models)

if (ESPEAK_INCLUDE_DIRS)
    set_source_files_properties(models/kokoro/phonemizer.cpp PROPERTIES COMPILE_FLAGS "${ESPEAK_CFLAGS_OTHER}")
    set_source_files_properties(models/kokoro/phonemizer.cpp PROPERTIES INCLUDE_DIRECTORIES "${ESPEAK_INCLUDE_DIRS}")
    target_link_libraries(tts PUBLIC ${ESPEAK_LIBRARIES})
endif ()
