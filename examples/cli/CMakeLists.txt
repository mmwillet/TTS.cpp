set(TARGET tts-cli)

add_executable(${TARGET}
    cli.cpp
    playback.cpp
    playback.h
    write_file.cpp
    write_file.h
    vad.cpp
    vad.h
)

find_package(SDL2)
if (SDL2_FOUND)
    target_link_libraries(${TARGET} PRIVATE SDL2::SDL2)
    set_source_files_properties(playback.cpp PROPERTIES COMPILE_FLAGS -DSDL2_INSTALL=1)
endif()

target_link_libraries(${TARGET} PRIVATE ggml tts)
