# sesolver_core: the pure numerical/physics core.
# HARD RULE: this target must NEVER depend on Qt, OpenGL/Vulkan, or any
# windowing / rendering library. Everything here is reachable by unit tests
# with analytic oracles (TDD loop: docs/TDD_RULES.md).
# placeholder.cpp exists only so the header-only static library links.
add_library(sesolver_core STATIC
    src/placeholder.cpp
)

target_include_directories(sesolver_core PUBLIC include)

# C++20 modules the core provides (consumers `import` these). Incremental
# migration off the header web: a leaf becomes a named module, and a consuming
# header may `import` it while still being #included elsewhere. PUBLIC so the
# app/test targets that link the core can import across the target boundary.
target_sources(sesolver_core
    PUBLIC FILE_SET CXX_MODULES FILES
        src/marching_cubes_tables.ixx
        src/colormap.ixx
        src/vram_budget.ixx
        src/flow.ixx
        src/cross_section.ixx
        src/camera.ixx
        src/volume.ixx)
target_compile_features(sesolver_core PUBLIC cxx_std_20)
target_link_libraries(sesolver_core PRIVATE ses_warnings)
target_link_libraries(sesolver_core PUBLIC ses_arch)

# OpenMP hot loops; pragmas are #ifdef-guarded so a non-OpenMP compiler still
# builds a correct serial core. PUBLIC: header-only code, consumers need the flag.
if(MSVC)
    # MSVC's default OpenMP 2.0 IGNORES collapse(2) (C4849); the LLVM runtime
    # implements modern OpenMP and links automatically.
    target_compile_options(sesolver_core PUBLIC /openmp:llvm)
    message(STATUS "OpenMP (MSVC LLVM runtime) - parallel core hot loops enabled.")
else()
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(sesolver_core PUBLIC OpenMP::OpenMP_CXX)
        message(STATUS "OpenMP found - parallel core hot loops enabled.")
    else()
        message(STATUS "OpenMP not found - core runs serial.")
    endif()
endif()
