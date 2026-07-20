# ---------------------------------------------------------------------------
# sesolver_app : the SDL3 GUI shell ("Humble Object") -- PURE presentation.
#
# This layer holds NO domain logic and NO solver code: it links the scenario
# library (physics orchestration), the viz library (raw-Vulkan renderer +
# swapchain presenter), and Dear ImGui, then owns only the window / input /
# main loop and the panel. Everything a demo IS lives in ses_scenario;
# everything the GPU computes lives in ses_solver. Verified manually plus the
# --selftest-* arcs (which run headless through ses_scenario).
#
# Reuse boundary (purist): SDL3 provides ONLY the window, input, and the
# Vulkan surface; Dear ImGui (vendored submodule) the immediate-mode UI.
# ---------------------------------------------------------------------------
find_package(SDL3 CONFIG REQUIRED)
find_package(volk CONFIG REQUIRED)  # imported targets are directory-scoped
# Command-line parsing is INFRASTRUCTURE, not physics: no wheel
# reinvention outside the numerical core (standing project rule).
find_package(Boost REQUIRED COMPONENTS program_options)

# ---------------------------------------------------------------------------
# Dear ImGui: the vendored submodule (external/imgui), compiled INTO the app.
# NOT the vcpkg port -- its vulkan-binding feature links the vulkan-1 import
# library, which is forbidden in the volk world (see tools/cmake/ses_vkfft.cmake).
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
# The app. Owns its two UI-side modules (panel + scheduler) privately.
# ---------------------------------------------------------------------------
add_executable(sesolver_app
    src/main.cpp
)
target_sources(sesolver_app
    PRIVATE FILE_SET CXX_MODULES FILES
        src/imgui_ui.ixx
        src/scheduler.ixx)
# GUI subsystem on Windows for optimized builds so double-clicking the exe
# does not spawn a companion console window (closing it would kill the app);
# SDL_main.h provides the WinMain adapter. Debug keeps the console so
# diagnostics stay visible outside a debugger.
set_target_properties(sesolver_app PROPERTIES
    WIN32_EXECUTABLE "$<CONFIG:Release,RelWithDebInfo,MinSizeRel>"
)
target_link_libraries(sesolver_app
    PRIVATE
        ses_scenario
        ses_viz
        SDL3::SDL3
        ses_imgui
        ses_warnings
        Boost::program_options
)
target_compile_features(sesolver_app PRIVATE cxx_std_20)
# _CRT_SECURE_NO_WARNINGS keeps MSVC quiet about std::getenv (the
# SES_VK_VALIDATION toggle read once at boot; race-free).
if(MSVC)
    target_compile_definitions(sesolver_app PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# ---------------------------------------------------------------------------
# Quantitative selftest arcs as OPT-IN ctest entries (review finding: the 23
# arcs verified real physics but lived outside ctest entirely). Default OFF:
# a full arc sweep is a multi-minute GPU battery (manifold alone ~3.5 min),
# far too heavy for the red/green TDD loop. Enable with -DSES_ARC_TESTS=ON,
# then run `ctest -L arc` (serialized: RESOURCE_LOCK gpu -- the project rule
# keeps the GPU exclusive during batteries).
# ---------------------------------------------------------------------------
option(SES_ARC_TESTS "Register the --selftest-* arcs as ctest entries (slow, GPU)" OFF)
if(SES_ARC_TESTS)
    foreach(_arc IN ITEMS anderson benzene billiard bloch cascade corral
            decay doubleslit2d
            dw1d efield energy h2p kepler ladder1d landau magnetic manifold
            morse1d partial pt1d qdot rabi scene trapdecay tunnel tunnel1d)
        add_test(NAME Arc.${_arc}
                 COMMAND sesolver_app --selftest-${_arc})
        set_tests_properties(Arc.${_arc} PROPERTIES
            LABELS "gpu;arc"
            RESOURCE_LOCK gpu
            TIMEOUT 600)
    endforeach()
    message(STATUS "Selftest arcs registered as ctest entries (ctest -L arc).")
endif()
