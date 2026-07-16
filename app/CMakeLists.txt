# ---------------------------------------------------------------------------
# sesolver_app : the SDL3 GUI shell ("Humble Object").
#
# This layer holds NO domain logic. It links sesolver_core, owns the window /
# input / main loop (SDL3), the Vulkan device (DeviceContext::create -- the
# same owning path sesolver_vkcheck exercises headless), the swapchain
# (ses.vk.present), and the Dear ImGui control panel. It is verified manually
# rather than by unit tests (the chosen Humble Object pattern) -- plus the
# --selftest-* arcs.
#
# Reuse boundary (purist): SDL3 provides ONLY the window, input, and the
# Vulkan surface; Dear ImGui (vendored submodule) the immediate-mode UI. All
# math, FFT, physics, and rendering logic are hand-written (ses_vk: volk +
# VMA + offline-baked SPIR-V).
# ---------------------------------------------------------------------------
find_package(SDL3 CONFIG REQUIRED)
find_package(VulkanHeaders CONFIG REQUIRED)  # vcpkg port: Vulkan::Headers
find_package(volk CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

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

# ---------------------------------------------------------------------------
# Slang AOT toolchain. Every ses_vk compute kernel and render shader is a Slang
# source baked to plain SPIR-V by slangc at build time and embedded as a C
# array via tools/cmake/bin2h.cmake (symbol k_<x>_spv). slangc is a prebuilt
# BUILD-TIME tool (FetchContent) -- it adds no runtime or vcpkg dependency;
# matrices are column-major to match the host std140 mat4 upload. (glslang
# stays, but ONLY as VkFFT's runtime shader compiler -- see ses_enable_vkfft.)
# ---------------------------------------------------------------------------
include(FetchContent)
set(SES_SLANG_VERSION "2026.13.1")
if(WIN32)
    set(_ses_slang_asset "slang-${SES_SLANG_VERSION}-windows-x86_64.zip")
else()
    set(_ses_slang_asset "slang-${SES_SLANG_VERSION}-linux-x86_64.zip")
endif()
FetchContent_Declare(ses_slang
    URL "https://github.com/shader-slang/slang/releases/download/v${SES_SLANG_VERSION}/${_ses_slang_asset}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(ses_slang)
find_program(SES_SLANGC slangc
    HINTS "${ses_slang_SOURCE_DIR}/bin"
    NO_DEFAULT_PATH REQUIRED)
message(STATUS "Slang AOT compiler: ${SES_SLANGC}")

set(_ses_vk_spv_dir "${CMAKE_CURRENT_BINARY_DIR}/vk_spv")
file(MAKE_DIRECTORY "${_ses_vk_spv_dir}")
set(_ses_vk_headers "")

# Shared Slang modules: every <name>.slang WITHOUT a stage extension (complex,
# ylm, grid, ...). They are `import`-ed by the shaders (via -I), never baked as
# entry points, and listed as bake deps so editing a module rebakes its
# consumers. CONFIGURE_DEPENDS re-globs when a module is added/removed.
set(_ses_shader_dir "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders")
file(GLOB _ses_shader_modules CONFIGURE_DEPENDS "${_ses_shader_dir}/*.slang")
list(FILTER _ses_shader_modules EXCLUDE REGEX "\\.(comp|vert|frag)\\.slang$")

# Bake one Slang shader (src/shaders/<srcfile>.slang) to a SPIR-V C header via
# slangc + bin2h (-DNAME=k_<x>_spv, header <x>_spv.h). The stage extension in
# the source name (.comp/.vert/.frag) is a human hint; slangc takes -stage.
function(ses_bake_shader _srcfile _stage _x)
    set(_slang "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${_srcfile}.slang")
    set(_spv "${_ses_vk_spv_dir}/${_x}.spv")
    set(_hdr "${_ses_vk_spv_dir}/${_x}_spv.h")
    add_custom_command(
        OUTPUT "${_hdr}"
        BYPRODUCTS "${_spv}"
        COMMAND "${SES_SLANGC}" "${_slang}"
                -I "${_ses_shader_dir}"
                -target spirv -entry main -stage ${_stage}
                -matrix-layout-column-major -o "${_spv}"
        COMMAND "${CMAKE_COMMAND}" "-DIN=${_spv}" "-DOUT=${_hdr}"
                "-DNAME=k_${_x}_spv"
                -P "${CMAKE_SOURCE_DIR}/tools/cmake/bin2h.cmake"
        DEPENDS "${_slang}" ${_ses_shader_modules}
                "${CMAKE_SOURCE_DIR}/tools/cmake/bin2h.cmake"
        COMMENT "slangc: ${_srcfile}.slang -> ${_x}_spv.h"
        VERBATIM)
    set(_hdrs "${_ses_vk_headers}")
    list(APPEND _hdrs "${_hdr}")
    set(_ses_vk_headers "${_hdrs}" PARENT_SCOPE)
endfunction()

# Compute kernels: source <k>.comp.slang, symbol k_<k>_spv.
foreach(_k IN ITEMS phase_multiply scale norm_peak norm_finalize scale_buf
        inner_product
        dipole_kick mean_force dipole pack_half unpack_half conj_scale
        shear axpy copy_state synth project_deposit
        bridge_store bridge_load flow_velocity
        accum bloom_down bloom_up compose particles
        occupancy occ_dilate shadow half_mul kin_mul damp_mul phase_damp_mul
        mcwf_axpy
        mc_density mc_classify mc_scan mc_emit
        fft_line8 fft_line64 fft_line256)
    ses_bake_shader("${_k}.comp" compute "${_k}")
endforeach()
# Render + presentation shader pairs (the blit pair feeds the shell's own
# swapchain pass). Dots in the stage extension are sanitized out of the symbol.
foreach(_k IN ITEMS mesh.vert mesh.frag volume.vert volume.frag
        slice.vert slice.frag
        flow.vert flow.frag blit.vert blit.frag)
    string(REPLACE "." "_" _x "${_k}")
    if(_k MATCHES "\\.vert$")
        set(_stage vertex)
    else()
        set(_stage fragment)
    endif()
    ses_bake_shader("${_k}" "${_stage}" "${_x}")
endforeach()
# One generation rule shared by every ses_vk consumer (the app's engine
# and the vkcheck harness) -- targets depend on this, not on the files.
add_custom_target(ses_spv_headers DEPENDS ${_ses_vk_headers})

# ---------------------------------------------------------------------------
# Dear ImGui: the vendored submodule (external/imgui), compiled INTO the app.
# NOT the vcpkg port -- its vulkan-binding feature links the vulkan-1 import
# library, which is forbidden in the volk world (see the NOTE above).
# IMGUI_IMPL_VULKAN_USE_VOLK makes the backend ride volk's loader instead.
# ---------------------------------------------------------------------------
set(SES_IMGUI_DIR "${CMAKE_SOURCE_DIR}/external/imgui")
if(NOT EXISTS "${SES_IMGUI_DIR}/imgui.cpp")
    message(FATAL_ERROR "external/imgui submodule missing -- run "
        "'git submodule update --init external/imgui'.")
endif()
add_library(ses_imgui STATIC
    "${SES_IMGUI_DIR}/imgui.cpp"
    "${SES_IMGUI_DIR}/imgui_draw.cpp"
    "${SES_IMGUI_DIR}/imgui_tables.cpp"
    "${SES_IMGUI_DIR}/imgui_widgets.cpp"
    "${SES_IMGUI_DIR}/backends/imgui_impl_sdl3.cpp"
    "${SES_IMGUI_DIR}/backends/imgui_impl_vulkan.cpp")
target_include_directories(ses_imgui SYSTEM PUBLIC
    "${SES_IMGUI_DIR}" "${SES_IMGUI_DIR}/backends")
target_compile_definitions(ses_imgui PUBLIC IMGUI_IMPL_VULKAN_USE_VOLK)
target_link_libraries(ses_imgui PUBLIC volk::volk SDL3::SDL3)
target_compile_features(ses_imgui PRIVATE cxx_std_20)

# ---------------------------------------------------------------------------
# ses_app_modules: the app layer's C++20 modules (ses.vk.* / ses.app.*). ONE
# owner target so the app and the vkcheck harness import a SINGLE BMI per
# module (per-target SES_HAVE_VKFFT would otherwise bake divergent engines).
# vma_impl.cpp is the one VMA implementation TU for the whole ses_vk world.
# ---------------------------------------------------------------------------
add_library(ses_app_modules STATIC
    src/vma_impl.cpp
)
target_sources(ses_app_modules
    PUBLIC FILE_SET CXX_MODULES FILES
        src/vk_device.ixx
        src/scheduler.ixx
        src/manifold_spec.ixx
        src/vk_compute.ixx
        src/scenario.ixx
        src/vram_probe.ixx
        src/vk_engine.ixx
        src/vk_present.ixx
        src/vk_render.ixx
        src/imgui_ui.ixx
        src/vk_blobs.ixx
        src/atom_model.ixx
        src/selftest_arcs.ixx
        src/base_director.ixx
        src/hydrogen_director.ixx
        src/harmonic_director.ixx
        src/tunneling_director.ixx)
target_compile_features(ses_app_modules PUBLIC cxx_std_20)
add_dependencies(ses_app_modules ses_spv_headers)
target_include_directories(ses_app_modules PRIVATE "${_ses_vk_spv_dir}")
# sesolver_core PUBLIC: app BMIs import core BMIs, and ses_arch codegen flags
# must reach lib and exes identically (BMI/importer mismatch = ODR trap).
# ses_imgui PUBLIC: imgui_ui.ixx's GMF needs <imgui.h> + the volk-riding
# backend define; static linking keeps windowing objects out of vkcheck (it
# never imports ses.app.* UI modules).
target_link_libraries(ses_app_modules
    PUBLIC
        sesolver_core
        volk::volk
        GPUOpen::VulkanMemoryAllocator
        Vulkan::Headers
        ses_imgui
    PRIVATE
        ses_warnings)
ses_enable_vkfft(ses_app_modules)
if(MSVC)
    target_compile_definitions(ses_app_modules PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# ---------------------------------------------------------------------------
# The app.
# ---------------------------------------------------------------------------
add_executable(sesolver_app
    src/main.cpp
)
# GUI subsystem on Windows for optimized builds so double-clicking the exe
# does not spawn a companion console window (closing it would kill the app);
# SDL_main.h provides the WinMain adapter. Debug keeps the console so
# diagnostics stay visible outside a debugger.
set_target_properties(sesolver_app PROPERTIES
    WIN32_EXECUTABLE "$<CONFIG:Release,RelWithDebInfo,MinSizeRel>"
)
add_dependencies(sesolver_app ses_spv_headers)
target_include_directories(sesolver_app PRIVATE "${_ses_vk_spv_dir}")
target_link_libraries(sesolver_app
    PRIVATE
        ses_app_modules
        sesolver_core
        SDL3::SDL3
        ses_imgui
        volk::volk
        GPUOpen::VulkanMemoryAllocator
        Vulkan::Headers
        ses_warnings
)
target_compile_features(sesolver_app PRIVATE cxx_std_20)
ses_enable_vkfft(sesolver_app)
# _CRT_SECURE_NO_WARNINGS keeps MSVC quiet about std::getenv (the
# SES_VK_VALIDATION toggle read once at boot; race-free).
if(MSVC)
    target_compile_definitions(sesolver_app PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# ---------------------------------------------------------------------------
# sesolver_vkcheck: THE GPU verification harness. NO windowing anywhere in
# this binary: volk loads the loader, VMA allocates, and the Slang kernel
# sources are baked to plain SPIR-V offline by slangc (ses_bake_shader) and
# embedded as C arrays via
# tools/cmake/bin2h.cmake. Every check pins a CPU double-precision reference
# value and tolerance. ctest: label "gpu", exit 77 = SKIP without a Vulkan
# device; SES_VK_VALIDATION=1 turns on the validation layers and fails the
# run on any layer error.
# ---------------------------------------------------------------------------
add_executable(sesolver_vkcheck src/vkcheck_main.cpp)
add_dependencies(sesolver_vkcheck ses_spv_headers)
# _CRT_SECURE_NO_WARNINGS keeps MSVC quiet about std::getenv (the
# SES_VK_VALIDATION toggle; reading it is race-free here).
if(MSVC)
    target_compile_definitions(sesolver_vkcheck PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
target_include_directories(sesolver_vkcheck PRIVATE "${_ses_vk_spv_dir}")
target_link_libraries(sesolver_vkcheck PRIVATE
    ses_app_modules
    sesolver_core
    volk::volk
    GPUOpen::VulkanMemoryAllocator
    ses_warnings)
target_compile_features(sesolver_vkcheck PRIVATE cxx_std_20)
# Native VkFFT: the plan runs on the engine's own device/queue, recorded
# straight into its primary command buffers (volk's canonical-name
# pointers satisfy header-only VkFFT; no vulkan-1 import lib -- see the
# volk NOTE above ses_enable_vkfft).
ses_enable_vkfft(sesolver_vkcheck)
add_test(NAME VkCheck COMMAND sesolver_vkcheck)
set_tests_properties(VkCheck PROPERTIES
    LABELS "gpu"
    SKIP_RETURN_CODE 77)
message(STATUS "Framework-free harness 'sesolver_vkcheck' enabled.")
