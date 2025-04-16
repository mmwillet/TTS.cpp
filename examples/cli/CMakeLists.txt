add_executable(cli
    cli.cpp
    write_file.cpp
    write_file.h
)

target_link_libraries(cli PRIVATE ggml tts)
