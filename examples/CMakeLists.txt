# examples

add_library(examples_common
        args.cpp
        args.h
        args_common.cpp
        args_common.h
        audio_file.h
)
target_include_directories(examples_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(examples_common PUBLIC ggml tts)

if (NOT EMSCRIPTEN)
    add_subdirectory(cli)
    add_subdirectory(perf_battery)
    add_subdirectory(quantize)
    add_subdirectory(server)
    add_subdirectory(phonemize)
endif ()
