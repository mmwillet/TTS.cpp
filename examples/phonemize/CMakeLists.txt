add_executable(phonemize phonemize.cpp)
target_link_libraries(phonemize PRIVATE ggml tts)
