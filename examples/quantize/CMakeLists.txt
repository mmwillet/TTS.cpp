add_executable(quantize quantize.cpp)
target_link_libraries(quantize PRIVATE ggml tts)

if (NOT WIN32)
    add_custom_target(quantize_snapshot_test.sh ALL DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/quantize_snapshot_test.sh)
    add_custom_command(OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/quantize_snapshot_test.sh
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/quantize_snapshot_test.sh
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/quantize_snapshot_test.sh
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/quantize_snapshot_test.sh)
    add_dependencies(quantize_snapshot_test.sh quantize)
endif ()
