# ---------------------------------------------------------------------------
# sesolver_app : the Qt + QRhi (Vulkan) GUI shell ("Humble Object").
#
# This layer holds NO domain logic. It links sesolver_core, owns the window /
# QRhiWidget (Qt) and the hand-written QRhi rendering + compute engine, and is
# verified manually rather than by unit tests (the chosen Humble Object
# pattern) -- plus the --selftest-* arcs.
#
# Reuse boundary (purist): Qt provides ONLY the window, the QRhi device layer,
# and UI. All math, FFT, physics, and rendering logic are hand-written.
# ---------------------------------------------------------------------------
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# The app itself now runs on QRhi/Vulkan: the private QRhi module, the qsb
# shader baker, and the Vulkan headers are REQUIRED (vcpkg manifest provides
# all three). NOTE: with the vcpkg Qt, Qt6::GuiPrivate must be requested by
# name -- find_package(Qt6 COMPONENTS Gui) alone does not create the target.
find_package(Qt6 REQUIRED COMPONENTS ShaderTools GuiPrivate)
find_package(VulkanHeaders CONFIG REQUIRED)  # vcpkg port: Vulkan::Headers

# VkFFT: header-only FFT + its runtime shader compiler (glslang). Optional --
# when absent the engine compiles without SES_HAVE_VKFFT and stays on the
# hand-rolled line FFT. VkFFT includes the bare 'glslang_c_interface.h';
# vcpkg installs it under glslang/Include/, so that directory must be on the
# include path. NOTE: every ses_vk target lives in the volk world
# (VK_NO_PROTOTYPES + PFN globals) -- NEVER link the vulkan-1 IMPORT library
# into one: its code thunks collide with volk's identically named function-
# pointer variables and volkInitialize crashes on startup.
find_package(glslang CONFIG QUIET)
find_path(SES_GLSLANG_C_INCLUDE_DIR glslang_c_interface.h
          PATH_SUFFIXES glslang/Include)
if(TARGET glslang::glslang AND SES_GLSLANG_C_INCLUDE_DIR)
    set(SES_VKFFT_AVAILABLE ON)
    message(STATUS "VkFFT enabled (engine step-body 3D transforms).")
else()
    set(SES_VKFFT_AVAILABLE OFF)
    message(STATUS "VkFFT unavailable -- engine keeps the hand-rolled FFT.")
endif()
function(ses_enable_vkfft target)
    if(SES_VKFFT_AVAILABLE)
        target_compile_definitions(${target} PRIVATE SES_HAVE_VKFFT=1 VKFFT_BACKEND=0)
        target_include_directories(${target} SYSTEM PRIVATE
            "${SES_GLSLANG_C_INCLUDE_DIR}")
        target_link_libraries(${target} PRIVATE
            glslang::glslang
            glslang::SPIRV
            glslang::glslang-default-resource-limits)
    endif()
endfunction()

add_executable(sesolver_app
    src/main.cpp
)
# GUI subsystem on Windows for optimized builds so double-clicking the exe
# does not spawn a companion console window (closing it would kill the app).
# Debug keeps the console so qCritical logs stay visible outside a
# debugger; fatal errors also surface via QMessageBox either way.
set_target_properties(sesolver_app PROPERTIES
    WIN32_EXECUTABLE "$<CONFIG:Release,RelWithDebInfo,MinSizeRel>"
)
target_link_libraries(sesolver_app
    PRIVATE
        sesolver_core
        Qt6::Core
        Qt6::Gui
        Qt6::GuiPrivate
        Qt6::Widgets
        Vulkan::Headers
        ses_warnings
)
target_compile_features(sesolver_app PRIVATE cxx_std_20)
ses_enable_vkfft(sesolver_app)

# qsb bakes ONLY the render shaders now (SPIR-V only: the app is Vulkan-
# backend). Every compute kernel reaches the ses_vk engine as offline-baked
# SPIR-V headers instead (the ses_spv_headers rule below).
qt6_add_shaders(sesolver_app "app_shaders"
    PREFIX "/shaders"
    NOGLSL NOHLSL NOMSL
    FILES
        src/qrhi_shaders/mesh.vert
        src/qrhi_shaders/mesh.frag
        src/qrhi_shaders/volume.vert
        src/qrhi_shaders/volume.frag
    OUTPUTS
        mesh.vert.qsb
        mesh.frag.qsb
        volume.vert.qsb
        volume.frag.qsb)

# Static Qt (the vcpkg x64-windows-static / x64-linux triplets) does NOT load
# plugins at runtime -- the platform plugin must be linked in, or the executable
# aborts with "no Qt platform plugin could be initialized". Collect the platform
# plugin (qwindows on Windows, xcb on Linux/X11) and import it into every GUI
# executable when Qt6 is a static build; a shared Qt loads plugins itself, so the
# guard makes this a no-op there.
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
endif()

# Qt DLLs must be findable when ctest launches the harness.
get_target_property(_qt6_core_dll Qt6::Core LOCATION)
get_filename_component(_qt6_bin_dir "${_qt6_core_dll}" DIRECTORY)

# ---------------------------------------------------------------------------
# sesolver_vkcheck: THE GPU verification harness (successor to gpucheck/GL
# and qrhicheck/QRhi, both retired). NO Qt anywhere in this binary: volk
# loads the loader, VMA allocates, and the Vulkan-GLSL kernel sources are
# baked to plain SPIR-V offline by glslangValidator (the glslang port's
# "tools" feature) and embedded as C arrays via tools/cmake/bin2h.cmake.
# Every check pins the same CPU double reference (and tolerance) the retired
# harnesses pinned -- the recorded values match digit for digit. ctest:
# label "gpu", exit 77 = SKIP without a Vulkan device; SES_VK_VALIDATION=1
# turns on the validation layers and fails the run on any layer error.
# ---------------------------------------------------------------------------
find_package(volk CONFIG QUIET)
find_package(VulkanMemoryAllocator CONFIG QUIET)
find_program(SES_GLSLANG_VALIDATOR glslangValidator
    HINTS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/tools/glslang")
if(TARGET volk::volk AND TARGET GPUOpen::VulkanMemoryAllocator
        AND SES_GLSLANG_VALIDATOR)
    set(_ses_vk_spv_dir "${CMAKE_CURRENT_BINARY_DIR}/vk_spv")
    file(MAKE_DIRECTORY "${_ses_vk_spv_dir}")
    set(_ses_vk_headers "")
    foreach(_k IN ITEMS phase_multiply scale norm_peak inner_product
            dipole_kick mean_force dipole pack_half unpack_half conj_scale
            shear axpy copy_state synth project_deposit
            bridge_store bridge_load
            fft_line8 fft_line64 fft_line256)
        set(_src "${CMAKE_CURRENT_SOURCE_DIR}/src/qrhi_shaders/${_k}.comp")
        set(_spv "${_ses_vk_spv_dir}/${_k}.spv")
        set(_hdr "${_ses_vk_spv_dir}/${_k}_spv.h")
        add_custom_command(
            OUTPUT "${_hdr}"
            BYPRODUCTS "${_spv}"
            COMMAND "${SES_GLSLANG_VALIDATOR}" -V --target-env vulkan1.1
                    -o "${_spv}" "${_src}"
            COMMAND "${CMAKE_COMMAND}" "-DIN=${_spv}" "-DOUT=${_hdr}"
                    "-DNAME=k_${_k}_spv"
                    -P "${CMAKE_SOURCE_DIR}/tools/cmake/bin2h.cmake"
            DEPENDS "${_src}" "${CMAKE_SOURCE_DIR}/tools/cmake/bin2h.cmake"
            COMMENT "glslangValidator: ${_k}.comp -> SPIR-V header"
            VERBATIM)
        list(APPEND _ses_vk_headers "${_hdr}")
    endforeach()
    # One generation rule shared by every ses_vk consumer (the app's engine
    # and the vkcheck harness) -- targets depend on this, not on the files.
    add_custom_target(ses_spv_headers DEPENDS ${_ses_vk_headers})

    # The app's compute engine is ses_vk (M5 Stage 9): raw Vulkan on the
    # QRhiWidget-adopted device, kernels from the offline-baked SPIR-V.
    add_dependencies(sesolver_app ses_spv_headers)
    target_include_directories(sesolver_app PRIVATE "${_ses_vk_spv_dir}")
    target_link_libraries(sesolver_app PRIVATE
        volk::volk GPUOpen::VulkanMemoryAllocator)

    add_executable(sesolver_vkcheck src/vkcheck_main.cpp)
    add_dependencies(sesolver_vkcheck ses_spv_headers)
    # A deliberately Qt-free binary: no moc/rcc/uic passes, and
    # _CRT_SECURE_NO_WARNINGS keeps MSVC quiet about std::getenv (the
    # SES_VK_VALIDATION toggle; reading it is race-free here).
    set_target_properties(sesolver_vkcheck PROPERTIES
        AUTOMOC OFF AUTORCC OFF AUTOUIC OFF)
    if(MSVC)
        target_compile_definitions(sesolver_vkcheck PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    target_include_directories(sesolver_vkcheck PRIVATE "${_ses_vk_spv_dir}")
    target_link_libraries(sesolver_vkcheck PRIVATE
        sesolver_core
        volk::volk
        GPUOpen::VulkanMemoryAllocator
        ses_warnings)
    target_compile_features(sesolver_vkcheck PRIVATE cxx_std_20)
    # Native VkFFT: the plan runs on the engine's own device/queue, recorded
    # straight into its primary command buffers (volk's canonical-name
    # pointers satisfy header-only VkFFT; no QRhi external blocks, and no
    # vulkan-1 import lib -- see ses_enable_vkfft's volk flavor).
    ses_enable_vkfft(sesolver_vkcheck volk)
    add_test(NAME VkCheck COMMAND sesolver_vkcheck)
    set_tests_properties(VkCheck PROPERTIES
        LABELS "gpu"
        SKIP_RETURN_CODE 77)
    message(STATUS "Framework-free harness 'sesolver_vkcheck' enabled.")
else()
    # The app's compute engine is ses_vk -- these are hard requirements now
    # (the vcpkg manifest provides all three on every platform).
    message(FATAL_ERROR "sesolver_app needs volk + vulkan-memory-allocator "
        "+ glslangValidator (vcpkg manifest deps; is glslang[tools] on?).")
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
