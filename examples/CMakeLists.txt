# examples

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (EMSCRIPTEN)
else()
    add_subdirectory(cli)
    add_subdirectory(perf_battery)
    add_subdirectory(quantize)
    add_subdirectory(server)
    add_subdirectory(phonemize)
endif()
