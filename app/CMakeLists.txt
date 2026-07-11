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

# VkFFT (M4): header-only FFT + its runtime shader compiler (glslang) + the
# Vulkan loader import lib. Optional -- when absent the engine compiles
# without SES_HAVE_VKFFT and stays on the hand-rolled line FFT. VkFFT
# includes the bare 'glslang_c_interface.h'; vcpkg installs it under
# glslang/Include/, so that directory must be on the include path.
find_package(Vulkan QUIET)
find_package(glslang CONFIG QUIET)
find_path(SES_GLSLANG_C_INCLUDE_DIR glslang_c_interface.h
          PATH_SUFFIXES glslang/Include)
if(TARGET Vulkan::Vulkan AND TARGET glslang::glslang AND SES_GLSLANG_C_INCLUDE_DIR)
    set(SES_VKFFT_AVAILABLE ON)
    message(STATUS "VkFFT enabled (engine step-body 3D transforms).")
else()
    set(SES_VKFFT_AVAILABLE OFF)
    message(STATUS "VkFFT unavailable -- engine keeps the hand-rolled FFT.")
endif()
# One helper applies the flags to any target embedding the engine header.
function(ses_enable_vkfft target)
    if(SES_VKFFT_AVAILABLE)
        target_compile_definitions(${target} PRIVATE SES_HAVE_VKFFT=1 VKFFT_BACKEND=0)
        target_include_directories(${target} SYSTEM PRIVATE
            "${SES_GLSLANG_C_INCLUDE_DIR}")
        target_link_libraries(${target} PRIVATE
            Vulkan::Vulkan
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

# Bake every compute kernel the QrhiEngine loads at 256^3 plus the mesh +
# volume render shaders into the app's resources (SPIR-V only: the app is
# Vulkan-backend, and the default ESSL targets reject compute shaders).
qt6_add_shaders(sesolver_app "app_shaders"
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
        src/qrhi_shaders/pack_half.comp
        src/qrhi_shaders/unpack_half.comp
        src/qrhi_shaders/conj_scale.comp
        src/qrhi_shaders/axpy.comp
        src/qrhi_shaders/copy_state.comp
        src/qrhi_shaders/shear.comp
        src/qrhi_shaders/synth.comp
        src/qrhi_shaders/bridge_store.comp
        src/qrhi_shaders/bridge_load.comp
        src/qrhi_shaders/project_deposit.comp
        src/qrhi_shaders/fft_line256.comp
        src/qrhi_shaders/mesh.vert
        src/qrhi_shaders/mesh.frag
        src/qrhi_shaders/volume.vert
        src/qrhi_shaders/volume.frag
    OUTPUTS
        phase_multiply.comp.qsb
        scale.comp.qsb
        norm_peak.comp.qsb
        inner_product.comp.qsb
        dipole_kick.comp.qsb
        mean_force.comp.qsb
        dipole.comp.qsb
        pack_half.comp.qsb
        unpack_half.comp.qsb
        conj_scale.comp.qsb
        axpy.comp.qsb
        copy_state.comp.qsb
        shear.comp.qsb
        synth.comp.qsb
        bridge_store.comp.qsb
        bridge_load.comp.qsb
        project_deposit.comp.qsb
        fft_line256.comp.qsb
        mesh.vert.qsb
        mesh.frag.qsb
        volume.vert.qsb
        volume.frag.qsb)

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
endif()

# Qt DLLs must be findable when ctest launches the harness.
get_target_property(_qt6_core_dll Qt6::Core LOCATION)
get_filename_component(_qt6_bin_dir "${_qt6_core_dll}" DIRECTORY)

# ---------------------------------------------------------------------------
# sesolver_qrhicheck: QRhi (Vulkan backend) verification harness -- the seed of
# the GL->QRhi migration (M1). Bakes the decorated Vulkan-GLSL kernels to SPIR-V
# via qsb (qt6_add_shaders) and runs them through QRhi, comparing against the
# same CPU double reference the core tests pin. Built only when the QRhi private
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
            src/qrhi_shaders/fft_line256.comp
            src/qrhi_shaders/conj_scale.comp
            src/qrhi_shaders/axpy.comp
            src/qrhi_shaders/copy_state.comp
            src/qrhi_shaders/shear.comp
            src/qrhi_shaders/synth.comp
            src/qrhi_shaders/bridge_store.comp
            src/qrhi_shaders/bridge_load.comp
            src/qrhi_shaders/project_deposit.comp
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
            fft_line8.comp.qsb
            fft_line256.comp.qsb
            conj_scale.comp.qsb
            axpy.comp.qsb
            copy_state.comp.qsb
            shear.comp.qsb
            synth.comp.qsb
            bridge_store.comp.qsb
            bridge_load.comp.qsb
            project_deposit.comp.qsb)
    # VkFFT: engine integration + the standalone spike check (see the shared
    # helper above; the check self-skips when the ports are absent).
    ses_enable_vkfft(sesolver_qrhicheck)
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

# ---------------------------------------------------------------------------
# sesolver_vkcheck: framework-free Vulkan verification harness (M5 Stage 0).
# NO Qt anywhere in this binary: volk loads the loader, VMA allocates, and the
# SAME Vulkan-GLSL kernel sources are baked to plain SPIR-V offline by
# glslangValidator (the glslang port's "tools" feature) and embedded as C
# arrays via tools/cmake/bin2h.cmake -- the qsb-independent shader path. Each
# kernel is verified against the SAME CPU double reference as qrhicheck, so
# the two harnesses cross-check as the M5 port proceeds. Skipped with a status
# message when volk/VMA/glslangValidator are absent. ctest: label "gpu",
# exit 77 = SKIP without a Vulkan device; SES_VK_VALIDATION=1 turns on the
# validation layers (classic-mode vcpkg install or SDK) and fails on errors.
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
    foreach(_k IN ITEMS phase_multiply)
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
    add_executable(sesolver_vkcheck src/vkcheck_main.cpp ${_ses_vk_headers})
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
    add_test(NAME VkCheck COMMAND sesolver_vkcheck)
    set_tests_properties(VkCheck PROPERTIES
        LABELS "gpu"
        SKIP_RETURN_CODE 77)
    message(STATUS "Framework-free harness 'sesolver_vkcheck' enabled.")
else()
    message(STATUS "sesolver_vkcheck skipped "
        "(needs volk + vulkan-memory-allocator + glslangValidator).")
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
