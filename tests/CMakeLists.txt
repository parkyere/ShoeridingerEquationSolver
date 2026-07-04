# ---------------------------------------------------------------------------
# Unit tests for sesolver_core (GoogleTest).
#
# GoogleTest is fetched at configure time (needs network on first configure;
# it is then cached under the build tree). Every behavior of the core library
# is driven from here, test-first (see docs/TDD_RULES.md).
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)
# Windows: make gtest use the same shared CRT as the rest of the build.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(sesolver_tests
    complex_test.cpp
    fft_test.cpp
    field_test.cpp
    grid_test.cpp
    harmonic_dynamics_test.cpp
    imaginary_time_test.cpp
    potential_test.cpp
    propagator_test.cpp
    spectral_test.cpp
    wavepacket_test.cpp
)
target_link_libraries(sesolver_tests
    PRIVATE
        sesolver_core
        GTest::gtest_main
)
target_compile_features(sesolver_tests PRIVATE cxx_std_20)

include(GoogleTest)
gtest_discover_tests(sesolver_tests)
