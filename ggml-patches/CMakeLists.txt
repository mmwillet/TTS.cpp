target_sources(ggml PRIVATE
        ggml-iterator.h
        llama-mmap.cpp
        llama-mmap.h
)
target_include_directories(ggml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
