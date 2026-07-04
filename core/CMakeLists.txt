# ---------------------------------------------------------------------------
# sesolver_core : the pure numerical/physics core.
#
# HARD RULE: this target must NEVER depend on Qt, OpenGL, or any windowing /
# rendering library. It contains only hand-written math, grid, field, FFT,
# operators, time propagation, potentials, and the pure-data side of
# visualization (e.g. marching-cubes geometry, transfer-function math).
# Everything here is reachable by unit tests with analytic oracles.
#
# Sources are added test-first through the TDD loop (see docs/TDD_RULES.md).
# `placeholder.cpp` only exists so the static library is linkable while the
# library is empty; delete it the moment the first real translation unit lands.
# ---------------------------------------------------------------------------
add_library(sesolver_core STATIC
    src/placeholder.cpp
)

target_include_directories(sesolver_core PUBLIC include)
target_compile_features(sesolver_core PUBLIC cxx_std_20)
target_link_libraries(sesolver_core PRIVATE ses_warnings)
target_link_libraries(sesolver_core PUBLIC ses_arch)

# Parallel hot loops (3D FFT lines, pointwise phase tables). The pragmas are
# guarded by #ifdef _OPENMP, so a compiler without OpenMP still builds a
# correct serial core. PUBLIC: the code lives in headers, so consumers need
# the flag too.
if(MSVC)
    # MSVC's default OpenMP is 2.0, which IGNORES collapse(2) (warning C4849).
    # The LLVM runtime implements modern OpenMP; it also links automatically.
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
