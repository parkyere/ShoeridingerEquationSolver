# ---------------------------------------------------------------------------
# sesolver_app : the Qt + OpenGL GUI shell ("Humble Object").
#
# This layer holds NO domain logic. It links sesolver_core, owns the window /
# OpenGL context (Qt) and the hand-written OpenGL rendering, and is verified
# manually rather than by unit tests (the chosen Humble Object pattern).
#
# Reuse boundary (purist): Qt provides ONLY the window, GL context, and UI.
# All math, FFT, physics, and the OpenGL rendering logic are hand-written.
# ---------------------------------------------------------------------------
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

add_executable(sesolver_app
    src/main.cpp
)
# GUI subsystem on Windows for optimized builds so double-clicking the exe
# does not spawn a companion console window (closing it would kill the app).
# Debug keeps the console so qCritical GL logs stay visible outside a
# debugger; fatal errors also surface via QMessageBox either way.
set_target_properties(sesolver_app PROPERTIES
    WIN32_EXECUTABLE "$<CONFIG:Release,RelWithDebInfo,MinSizeRel>"
)
target_link_libraries(sesolver_app
    PRIVATE
        sesolver_core
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
        Qt6::OpenGL
        Qt6::OpenGLWidgets
        ses_warnings
)
target_compile_features(sesolver_app PRIVATE cxx_std_20)
