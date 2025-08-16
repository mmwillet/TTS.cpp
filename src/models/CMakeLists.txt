target_sources(tts PRIVATE
        loaders.cpp
        loaders.h
)

add_subdirectory(dia)
add_subdirectory(dummy)
add_subdirectory(kokoro)
add_subdirectory(orpheus)
add_subdirectory(parler)
