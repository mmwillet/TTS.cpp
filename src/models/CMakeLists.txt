target_sources(tts PRIVATE
        loaders.cpp
        loaders.h
)

add_subdirectory(dia)
if (LINUX)
    add_subdirectory(dummy)
endif ()
add_subdirectory(kokoro)
add_subdirectory(orpheus)
add_subdirectory(parler)
