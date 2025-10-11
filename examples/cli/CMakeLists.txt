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

option(TTS_CLI_SDL    "TTS.cpp: build the example cli with SDL" ON)

if(TTS_CLI_SDL)
    find_package(SDL2 QUIET)
    if (NOT SDL2_FOUND)
        include(FetchContent)
        message(STATUS "Could not find SDL2, fetching from git...")
        FetchContent_Declare(
            SDL2
            GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
            GIT_TAG        release-2.32.8
        )
        FetchContent_MakeAvailable(SDL2)
    endif()

    target_link_libraries(${TARGET} PRIVATE SDL2::SDL2)
    set_source_files_properties(playback.cpp PROPERTIES COMPILE_FLAGS -DSDL2_INSTALL=1)
endif()

target_link_libraries(${TARGET} PRIVATE ggml tts)
install(TARGETS ${TARGET} RUNTIME)
