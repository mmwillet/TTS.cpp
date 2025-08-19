add_executable(aPaleBlueDot-test
        main.cpp
)
target_link_libraries(aPaleBlueDot-test tts)
# The use valgrind --tool=massif
