if (WHISPER_SDL2)
    # talk-server
    set(TARGET talk-server)
    add_executable(${TARGET} talk-server.cpp)
    target_link_libraries(${TARGET} PRIVATE common common-sdl curl whisper ${CMAKE_THREAD_LIBS_INIT})

    include(DefaultTargetOptions)
endif ()
