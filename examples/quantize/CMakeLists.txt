add_executable(quantize
        quantize.cpp
        quantize_impl.cpp
        quantize_impl.h
)
target_link_libraries(quantize PRIVATE ggml tts)
