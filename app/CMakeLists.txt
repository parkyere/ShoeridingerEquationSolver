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

# Static Qt (the vcpkg x64-windows-static / x64-linux triplets) does NOT load
# plugins at runtime -- the platform plugin must be linked in, or the executable
# aborts with "no Qt platform plugin could be initialized". Collect the platform
# plugin (qwindows on Windows, xcb on Linux/X11) and import it into every GUI
# executable when Qt6 is a static build; a shared Qt loads plugins itself, so the
# guard makes this a no-op there. Reused by the sesolver_qrhicheck block below.
get_target_property(_ses_qt_core_type Qt6::Core TYPE)
set(_ses_qt_platform_plugins "")
if(_ses_qt_core_type STREQUAL "STATIC_LIBRARY")
    if(TARGET Qt6::QWindowsIntegrationPlugin)
        list(APPEND _ses_qt_platform_plugins Qt6::QWindowsIntegrationPlugin)
    endif()
    if(TARGET Qt6::QXcbIntegrationPlugin)
        list(APPEND _ses_qt_platform_plugins Qt6::QXcbIntegrationPlugin)
    endif()
endif()
if(_ses_qt_platform_plugins)
    qt6_import_plugins(sesolver_app INCLUDE ${_ses_qt_platform_plugins})
    qt6_import_plugins(sesolver_gpucheck INCLUDE ${_ses_qt_platform_plugins})
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

# ---------------------------------------------------------------------------
# sesolver_qrhicheck: QRhi (Vulkan backend) verification harness -- the seed of
# the GL->QRhi migration (M1). Bakes the decorated Vulkan-GLSL kernels to SPIR-V
# via qsb (qt6_add_shaders) and runs them through QRhi, comparing against the
# same CPU reference as sesolver_gpucheck. Built only when the QRhi private
# module (Qt6::GuiPrivate), the shader baker (Qt6::ShaderTools), and the vcpkg
# vulkan-headers (Vulkan::Headers, needed to compile QVulkanInstance) are all
# present; otherwise skipped with a status message so the normal build is
# unaffected. Registered with ctest (label "gpu", exit 77 = SKIP when no device).
# ---------------------------------------------------------------------------
# ShaderTools and GuiPrivate are each requested explicitly (separate calls so a
# missing one does not silently void the other). NOTE: with the vcpkg Qt,
# find_package(Qt6 COMPONENTS Gui) alone does NOT create the Qt6::GuiPrivate
# target -- it must be requested by name, or sesolver_qrhicheck silently skips.
find_package(Qt6 QUIET COMPONENTS ShaderTools)
find_package(Qt6 QUIET COMPONENTS GuiPrivate)
find_package(VulkanHeaders CONFIG QUIET)  # vcpkg port: provides Vulkan::Headers
if(TARGET Qt6::GuiPrivate AND TARGET Qt6::ShaderTools AND TARGET Vulkan::Headers)
    add_executable(sesolver_qrhicheck src/qrhicheck_main.cpp)
    # Vulkan::Headers (vcpkg vulkan-headers) puts vulkan.h on a SYSTEM include
    # path so QVulkanInstance compiles (and -Wpedantic stays quiet inside it);
    # the runtime uses the vulkan-1.dll loader Qt dlopens.
    target_link_libraries(sesolver_qrhicheck
        PRIVATE sesolver_core Qt6::Core Qt6::Gui Qt6::GuiPrivate Vulkan::Headers ses_warnings)
    target_compile_features(sesolver_qrhicheck PRIVATE cxx_std_20)
    # SPIR-V only: this harness targets the Vulkan backend, and the default
    # GLSL cross-compile targets (ESSL 100/120) reject compute shaders. qsb
    # always embeds SPIR-V, which is what QRhi's Vulkan backend consumes.
    qt6_add_shaders(sesolver_qrhicheck "qrhicheck_shaders"
        PREFIX "/shaders"
        NOGLSL NOHLSL NOMSL
        FILES
            src/qrhi_shaders/phase_multiply.comp
            src/qrhi_shaders/scale.comp
            src/qrhi_shaders/norm_peak.comp
            src/qrhi_shaders/inner_product.comp
            src/qrhi_shaders/dipole_kick.comp
            src/qrhi_shaders/mean_force.comp
            src/qrhi_shaders/dipole.comp
            src/qrhi_shaders/fft_line64.comp
            src/qrhi_shaders/pack_half.comp
            src/qrhi_shaders/unpack_half.comp
            src/qrhi_shaders/fft_line8.comp
        OUTPUTS
            phase_multiply.comp.qsb
            scale.comp.qsb
            norm_peak.comp.qsb
            inner_product.comp.qsb
            dipole_kick.comp.qsb
            mean_force.comp.qsb
            dipole.comp.qsb
            fft_line64.comp.qsb
            pack_half.comp.qsb
            unpack_half.comp.qsb
            fft_line8.comp.qsb)
    if(_ses_qt_platform_plugins)
        qt6_import_plugins(sesolver_qrhicheck INCLUDE ${_ses_qt_platform_plugins})
    endif()
    add_test(NAME QrhiCheck COMMAND sesolver_qrhicheck)
    set_tests_properties(QrhiCheck PROPERTIES
        LABELS "gpu"
        SKIP_RETURN_CODE 77
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${_qt6_bin_dir}"
    )
    message(STATUS "QRhi migration harness 'sesolver_qrhicheck' enabled.")
else()
    message(STATUS "sesolver_qrhicheck skipped "
        "(needs Qt6::GuiPrivate + Qt6::ShaderTools + Vulkan::Headers via vcpkg).")
endif()

# Stage the Qt runtime next to the app after every build, so it launches
# from the IDE or Explorer without Qt on PATH (the classic missing-
# Qt6Widgets.dll failure when pressing F5 in Visual Studio). Skipped for a
# static Qt build -- there are no Qt DLLs to deploy.
if(WIN32 AND NOT _ses_qt_core_type STREQUAL "STATIC_LIBRARY")
    find_program(SES_WINDEPLOYQT windeployqt HINTS "${_qt6_bin_dir}")
    if(SES_WINDEPLOYQT)
        add_custom_command(TARGET sesolver_app POST_BUILD
            COMMAND "${SES_WINDEPLOYQT}" --no-translations
                    "$<TARGET_FILE:sesolver_app>"
            COMMENT "windeployqt: staging the Qt runtime next to sesolver_app")
    endif()
endif()
