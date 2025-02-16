# TODO: should not use this
if (WIN32)
    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

set(ESPEAK_INSTALL_DIR "$ENV{ESPEAK_INSTALL_DIR}")

if (ESPEAK_INSTALL_DIR)
    set(ESPEAK_LIB_DIR ${ESPEAK_INSTALL_DIR}/lib)
    set(ESPEAK_INCLUDE_DIR ${ESPEAK_INSTALL_DIR}/include/espeak-ng)
endif()

# parler

add_library(parler
            ../include/parler.h
            ../include/args.h
            ../include/phonemizer.h
            parler.cpp
            tokenizer.cpp
            sampler.cpp
            dac_runner.cpp
            parler_gguf.cpp
            parler_model.cpp
            dac_model.cpp
            util.cpp
            args.cpp
            t5_encoder_model.cpp
            phonemizer.cpp
            tts_model.cpp
            )

target_include_directories(parler PUBLIC . ../include ../ggml/src/)

if (ESPEAK_INSTALL_DIR)
    target_include_directories(parler PUBLIC . "${ESPEAK_INCLUDE_DIR}")
    add_definitions(-DESPEAK_DATA_PATH=\"${ESPEAK_INSTALL_DIR}/share\")
endif()

target_compile_features   (parler PUBLIC cxx_std_11) # don't bump

target_link_libraries(parler PUBLIC ggml)

if (ESPEAK_INSTALL_DIR)
    if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        target_link_libraries(parler PUBLIC "${ESPEAK_LIB_DIR}/libespeak-ng.dylib")
    elseif(WIN32)
        target_link_libraries(parler PUBLIC "${ESPEAK_LIB_DIR}/libespeak-ng.dll")
    else()
        target_link_libraries(parler PUBLIC "${ESPEAK_LIB_DIR}/libespeak-ng.o")
    endif()
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(parler PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(parler PRIVATE LLAMA_BUILD)
    target_compile_definitions(parler PUBLIC  LLAMA_SHARED)
endif()
