cmake_minimum_required(VERSION 3.21)

# ---------------------------------------------------------------------------
# Optional: pick the newest installed clang++ / g++ BEFORE the compiler is
# locked, so a Linux "just press F5" run gets a recent LLVM/GCC without picking
# a kit by hand. Opt-in with -DSES_AUTODETECT_COMPILER=ON (the `linux-native`
# preset sets it); skipped if you already pinned CMAKE_CXX_COMPILER or set $CXX;
# falls through safely to CMake's default if nothing on the list is found.
# ---------------------------------------------------------------------------
# NOTE: UNIX/WIN32 are only set after project(); use CMAKE_HOST_UNIX (available
# from startup) since this is about the build host's compilers.
if(CMAKE_HOST_UNIX AND SES_AUTODETECT_COMPILER
   AND NOT DEFINED CMAKE_CXX_COMPILER AND "$ENV{CXX}" STREQUAL "")
    # Newest first (LLVM ahead of GCC). Extend the top as toolchains land.
    foreach(_cxx IN ITEMS
            clang++-25 clang++-24 clang++-23 clang++-22 clang++-21 clang++-20
            clang++-19 clang++-18 clang++-17 clang++-16
            g++-16 g++-15 g++-14 g++-13 g++-12 clang++ g++)
        # unset(CACHE) each iteration: find_program caches its result, so a
        # reused var would keep the first (possibly NOTFOUND) answer forever.
        unset(_ses_cxx_path CACHE)
        find_program(_ses_cxx_path NAMES ${_cxx})
        if(_ses_cxx_path)
            set(CMAKE_CXX_COMPILER "${_ses_cxx_path}" CACHE FILEPATH "C++ compiler" FORCE)
            string(REPLACE "clang++" "clang" _cc "${_cxx}")
            string(REPLACE "g++" "gcc" _cc "${_cc}")
            unset(_ses_cc_path CACHE)
            find_program(_ses_cc_path NAMES ${_cc})
            if(_ses_cc_path)
                set(CMAKE_C_COMPILER "${_ses_cc_path}" CACHE FILEPATH "C compiler" FORCE)
            endif()
            message(STATUS "SES_AUTODETECT_COMPILER: using ${CMAKE_CXX_COMPILER}")
            break()
        endif()
    endforeach()
endif()

project(ShoeridingerEquationSolver
    VERSION 0.0.1
    DESCRIPTION "From-scratch FDM/TDSE single-electron Schrodinger solver with framework-free Vulkan compute and rendering in an SDL3 window shell"
    LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Language standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# C++20 named modules: scan every C++20 TU for `import` so consumers (which only
# import, not provide) resolve modules the core provides. Required because a
# consumed module forces all consuming targets to be scanned, and the project
# baseline (cmake_minimum_required 3.21) leaves CMP0155 OLD (scan off).
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

# Single-config generators: default to Debug for the TDD loop.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# compile_commands.json for clangd / tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(SES_BUILD_TESTS        "Build unit tests"                          ON)
option(SES_BUILD_APP          "Build the SDL3/Vulkan app (requires SDL3)" ON)
option(SES_BUILD_BENCH        "Build the manual micro-benchmark"          ON)
option(SES_ENABLE_AVX2        "Compile with AVX2 (x86-64 CPUs from ~2013)" ON)
option(SES_NATIVE             "Tune for the build host CPU (-march=native)" OFF)
option(SES_AUTODETECT_COMPILER "Pre-project: use the newest clang++/g++ on PATH" OFF)
option(SES_WARNINGS_AS_ERRORS "Treat compiler warnings as errors"         OFF)

# ---------------------------------------------------------------------------
# Shared warning flags (INTERFACE target).
# NOTE: never enable -ffast-math / /fp:fast - it breaks NaN handling and
# bitwise reproducibility, which are load-bearing for numerical correctness.
# ---------------------------------------------------------------------------
add_library(ses_warnings INTERFACE)
if(MSVC)
    target_compile_options(ses_warnings INTERFACE /W4 /permissive- /EHsc)
    if(SES_WARNINGS_AS_ERRORS)
        target_compile_options(ses_warnings INTERFACE /WX)
    endif()
else()
    target_compile_options(ses_warnings INTERFACE -Wall -Wextra -Wpedantic)
    if(SES_WARNINGS_AS_ERRORS)
        target_compile_options(ses_warnings INTERFACE -Werror)
    endif()
endif()

# ---------------------------------------------------------------------------
# CPU architecture flags (INTERFACE target, linked PUBLIC by the core so every
# TU that compiles the header-only hot loops gets it).
#
# NOTE -ffp-contract=off: with bare -mavx2 -mfma the compiler may contract
# a + b*c into fma(b, c, a), changing rounding and breaking the exact-value
# FP oracles the tests pin (e.g. symmetric-grid coord(50) == 0.0). Contraction
# off keeps scalar semantics BITWISE identical to the SSE2 build -- we take
# only the 4-wide double vectorization.
# ---------------------------------------------------------------------------
add_library(ses_arch INTERFACE)
if(NOT MSVC)
    # ses::Complex is std::complex: compile complex * and / to the naive
    # formulas (no C99 Annex G NaN/inf fixups, no __muldc3 branches in the
    # FFT hot loop) -- bitwise identical to the arithmetic the exact-value
    # test oracles were pinned against. (MSVC never emits Annex G fixups.)
    # GCC spells the flag -fcx-limited-range; Clang 18+ spells it
    # -fcomplex-arithmetic=basic; older Clang has neither, which is SAFE to
    # omit: __mulsc3 computes the same naive formula for finite values and
    # only patches NaN/inf afterward, so the oracles still hold -- the flag
    # only removes the libcall from the hot loop.
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-fcx-limited-range" SES_HAS_FCX_LIMITED_RANGE)
    if(SES_HAS_FCX_LIMITED_RANGE)
        target_compile_options(ses_arch INTERFACE -fcx-limited-range)
    else()
        check_cxx_compiler_flag("-fcomplex-arithmetic=basic"
                                SES_HAS_FCOMPLEX_ARITH_BASIC)
        if(SES_HAS_FCOMPLEX_ARITH_BASIC)
            target_compile_options(ses_arch INTERFACE -fcomplex-arithmetic=basic)
        else()
            message(STATUS "complex limited-range flag unsupported -- "
                "std::complex keeps Annex G (finite results identical).")
        endif()
    endif()
endif()
# SES_NATIVE (host-CPU tuning) takes precedence over the portable AVX2 flags.
# It keeps -ffp-contract=off AND pins the vector width to 256 (see below), so
# even on an AVX-512 host the auto-vectorized reductions match the AVX2 build
# bitwise -- the exact-value test oracles stay valid, and SES_NATIVE is safe for
# the app AND the suite. (If some toolchain still trips an oracle, build the
# suite with SES_NATIVE=OFF, the plain `linux` preset.)
if(MSVC)
    if(SES_ENABLE_AVX2 OR SES_NATIVE)
        target_compile_options(ses_arch INTERFACE /arch:AVX2)
        message(STATUS "AVX2 enabled (MSVC /arch:AVX2).")
    endif()
elseif(SES_NATIVE)
    # -march=native for host tuning, but PIN the vector width to 256 (4-wide
    # double = the AVX2 build) so auto-vectorized reductions keep the same
    # summation order/rounding on an AVX-512 host. On Zen4/znver4 this is also
    # the arch default (its AVX-512 is 256-bit double-pumped). -ffp-contract=off
    # keeps FMA fusion out, exactly as the AVX2 path does.
    target_compile_options(ses_arch INTERFACE
        -march=native -mprefer-vector-width=256 -ffp-contract=off)
    message(STATUS "Native CPU tuning (-march=native, 256-bit vectors).")
elseif(SES_ENABLE_AVX2)
    target_compile_options(ses_arch INTERFACE -mavx2 -mfma -ffp-contract=off)
    message(STATUS "AVX2 enabled (SES_ENABLE_AVX2=ON).")
endif()

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
# Pure, dependency-free numerical core. Always built.
add_subdirectory(core)

if(SES_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SES_BUILD_BENCH)
    add_subdirectory(bench)
endif()

# GUI shell. Auto-skipped (with a warning) when SDL3 is not installed, so the
# core library and the TDD test loop never depend on a windowing toolkit.
if(SES_BUILD_APP)
    find_package(SDL3 CONFIG QUIET)
    if(SDL3_FOUND)
        message(STATUS "SDL3 ${SDL3_VERSION} found - building 'sesolver_app'.")
        add_subdirectory(app)
    else()
        message(WARNING
            "SDL3 not found - skipping the GUI target 'sesolver_app'. "
            "The core library and tests still build. "
            "Install SDL3 (vcpkg manifest) and reconfigure to build the GUI.")
    endif()
endif()
