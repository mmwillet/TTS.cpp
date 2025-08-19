
cmake_minimum_required(VERSION 3.14)
project("tts.cpp" C CXX)
include(CheckIncludeFileCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_WARN_UNUSED_CLI YES)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (DEFINED ENV{ESPEAK_INSTALL_DIR})
    set(ESPEAK_INCLUDE_DIRS $ENV{ESPEAK_INSTALL_DIR}/include/espeak-ng)
    if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(ESPEAK_LIBRARIES "$ENV{ESPEAK_INSTALL_DIR}/lib/libespeak-ng.dylib")
    elseif(WIN32)
        set(ESPEAK_LIBRARIES "$ENV{ESPEAK_INSTALL_DIR}/lib/libespeak-ng.dll")
    else()
        set(ESPEAK_LIBRARIES "$ENV{ESPEAK_INSTALL_DIR}/lib/libespeak-ng.o")
    endif()
    set(ESPEAK_CFLAGS_OTHER "${ESPEAK_CFLAGS_OTHER} -DESPEAK_INSTALL=1 -DESPEAK_INSTALL_LOCAL=1 -DESPEAK_DATA_PATH=\\\"$ENV{ESPEAK_INSTALL_DIR}/share\\\"")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Tested with libespeak-ng-dev=1.51+dfsg-12build1
    find_package(PkgConfig)
    if (PKG_CONFIG_FOUND)
        pkg_check_modules(ESPEAK espeak-ng)
        if (ESPEAK_FOUND)
            set(ESPEAK_CFLAGS_OTHER "${ESPEAK_CFLAGS_OTHER} -DESPEAK_INSTALL=1 -DESPEAK_DATA_PATH=\\\"${ESPEAK_LIBDIR}/espeak-ng-data\\\"")
        endif()
    endif()
endif()

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Add path to modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(git-vars)

option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/source-charset:utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/source-charset:utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/execution-charset:utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/execution-charset:utf-8>")
endif()

# debug
option(TTS_ALL_WARNINGS           "TTS.cpp: enable all compiler warnings"                   ON)
option(TTS_ALL_WARNINGS_3RD_PARTY "TTS.cpp: enable all compiler warnings in 3rd party libs" OFF)

# build
option(TTS_FATAL_WARNINGS "TTS.cpp: enable -Werror flag" OFF)

# extra artifacts
option(TTS_BUILD_EXAMPLES "TTS.cpp: build examples" ON)

# Required for relocatable CMake package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)

# override ggml options
set(GGML_ALL_WARNINGS       ${TTS_ALL_WARNINGS})
set(GGML_FATAL_WARNINGS     ${TTS_FATAL_WARNINGS})


# build lib
add_subdirectory(ggml)
add_subdirectory(ggml-patches)
add_subdirectory(src)

# install tts

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(TTS_BUILD_NUMBER        ${BUILD_NUMBER})
set(TTS_BUILD_COMMIT        ${BUILD_COMMIT})
set(TTS_INSTALL_VERSION 0.0.${BUILD_NUMBER})

set(TTS_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(TTS_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(TTS_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")

# form copied from llama.cpp see explanation there.
set(GGML_TRANSIENT_DEFINES)
get_target_property(GGML_DIRECTORY ggml SOURCE_DIR)
get_directory_property(GGML_DIR_DEFINES DIRECTORY ${GGML_DIRECTORY} COMPILE_DEFINITIONS)
if (GGML_DIR_DEFINES)
    list(APPEND GGML_TRANSIENT_DEFINES ${GGML_DIR_DEFINES})
endif()
get_target_property(GGML_TARGET_DEFINES ggml COMPILE_DEFINITIONS)
if (GGML_TARGET_DEFINES)
    list(APPEND GGML_TRANSIENT_DEFINES ${GGML_TARGET_DEFINES})
endif()
get_target_property(GGML_LINK_LIBRARIES ggml LINK_LIBRARIES)

# all public headers
set(TTS_PUBLIC_HEADERS
    include/common.h
    include/tts.h
    include/audio_file.h
    include/args.h
    include/phonemizer.h)

set_target_properties(tts PROPERTIES PUBLIC_HEADER "${TTS_PUBLIC_HEADERS}")
install(TARGETS tts LIBRARY PUBLIC_HEADER)

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tts-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/tts-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tts
    PATH_VARS TTS_INCLUDE_INSTALL_DIR
              TTS_LIB_INSTALL_DIR
              TTS_BIN_INSTALL_DIR )

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/tts-version.cmake
    VERSION ${TTS_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/tts-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/parer-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tts)

configure_file(cmake/tts.pc.in
        "${CMAKE_CURRENT_BINARY_DIR}/tts.pc"
        @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/tts.pc"
        DESTINATION lib/pkgconfig)

# examples

if (TTS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

add_subdirectory(tests)
