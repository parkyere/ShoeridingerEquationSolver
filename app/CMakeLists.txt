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

# ---------------------------------------------------------------------------
# sesolver_gpucheck: verification harness for GPU compute kernels
# (docs/GPU_PLAN.md). Compares each kernel against the unit-tested CPU core;
# registered with ctest (label "gpu", exit 77 = SKIP when no 4.3 context).
# ---------------------------------------------------------------------------
add_executable(sesolver_gpucheck
    src/gpucheck_main.cpp
)
target_link_libraries(sesolver_gpucheck
    PRIVATE
        sesolver_core
        Qt6::Core
        Qt6::Gui
        Qt6::OpenGL
        ses_warnings
)
target_compile_features(sesolver_gpucheck PRIVATE cxx_std_20)

# Static Qt (e.g. the vcpkg x64-linux triplet) does NOT load plugins at
# runtime -- the platform plugin must be linked in, or the executable aborts
# with "no Qt platform plugin could be initialized". Import xcb (X11) for both
# GUI executables when Qt6 is a static build; shared Qt loads plugins itself,
# so both guards make this a no-op there (Windows MinGW/MSVC).
get_target_property(_ses_qt_core_type Qt6::Core TYPE)
if(_ses_qt_core_type STREQUAL "STATIC_LIBRARY" AND TARGET Qt6::QXcbIntegrationPlugin)
    qt6_import_plugins(sesolver_app INCLUDE Qt6::QXcbIntegrationPlugin)
    qt6_import_plugins(sesolver_gpucheck INCLUDE Qt6::QXcbIntegrationPlugin)
endif()

# Qt DLLs must be findable when ctest launches the harness.
get_target_property(_qt6_core_dll Qt6::Core LOCATION)
get_filename_component(_qt6_bin_dir "${_qt6_core_dll}" DIRECTORY)
add_test(NAME GpuCheck COMMAND sesolver_gpucheck)
set_tests_properties(GpuCheck PROPERTIES
    LABELS "gpu"
    SKIP_RETURN_CODE 77
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${_qt6_bin_dir}"
)

# Stage the Qt runtime next to the app after every build, so it launches
# from the IDE or Explorer without Qt on PATH (the classic missing-
# Qt6Widgets.dll failure when pressing F5 in Visual Studio).
if(WIN32)
    find_program(SES_WINDEPLOYQT windeployqt HINTS "${_qt6_bin_dir}")
    if(SES_WINDEPLOYQT)
        add_custom_command(TARGET sesolver_app POST_BUILD
            COMMAND "${SES_WINDEPLOYQT}" --no-translations
                    "$<TARGET_FILE:sesolver_app>"
            COMMENT "windeployqt: staging the Qt runtime next to sesolver_app")
    endif()
endif()
