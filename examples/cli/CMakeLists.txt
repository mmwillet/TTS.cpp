add_executable(cli
    cli.cpp
    playback.cpp
    playback.h
    write_file.cpp
    write_file.h
)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PULSE libpulse-simple)
    if (PULSE_FOUND)
        set_source_files_properties(playback.cpp PROPERTIES COMPILE_FLAGS "${PULSE_CFLAGS_OTHER} -DPULSE_INSTALL=1")
        set_source_files_properties(playback.cpp PROPERTIES INCLUDE_DIRECTORIES "${PULSE_INCLUDE_DIRS}")
        target_link_libraries(cli PRIVATE ${PULSE_LIBRARIES})
    endif()
endif()

target_link_libraries(cli PRIVATE ggml tts)
