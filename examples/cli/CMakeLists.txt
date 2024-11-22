add_executable(cli parler_cli.cpp)
target_link_libraries(cli PRIVATE ggml parler)
