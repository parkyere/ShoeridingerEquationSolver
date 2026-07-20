# ---------------------------------------------------------------------------
# Unit tests for sesolver_core (GoogleTest).
#
# GoogleTest: prefer a vcpkg-provided package (the hermetic vcpkg-unified path);
# fall back to fetching it at configure time for the non-vcpkg kit presets. Both
# providers expose the same GTest::gtest_main target used below. Every behavior
# of the core library is driven from here, test-first (see docs/TDD_RULES.md).
# ---------------------------------------------------------------------------
find_package(GTest CONFIG QUIET)
if(NOT TARGET GTest::gtest_main)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.2
    )
    # Windows: make gtest use the same shared CRT as the rest of the build.
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(sesolver_tests
    anderson1d_test.cpp
    billiard2d_test.cpp
    bloch_test.cpp
    camera_test.cpp
    colormap_test.cpp
    complex_test.cpp
    corral2d_test.cpp
    cross_section_test.cpp
    decay_test.cpp
    deflation_test.cpp
    degenerate_guards_test.cpp
    dipole_angular_test.cpp
    drive_test.cpp
    dynamics3_test.cpp
    eigenstate_flush_test.cpp
    emission_test.cpp
    fft3_test.cpp
    fft_test.cpp
    field3_test.cpp
    field_test.cpp
    flow_test.cpp
    grid3_test.cpp
    grid_test.cpp
    harmonic_dynamics_test.cpp
    harmonics_test.cpp
    heightfield_test.cpp
    hydrogen3_test.cpp
    imaginary_tables_test.cpp
    imaginary_time_test.cpp
    kepler_test.cpp
    ladder_test.cpp
    lattice2d_test.cpp
    magnetic_test.cpp
    marching_cubes_test.cpp
    mcwf_test.cpp
    measurement_test.cpp
    molecule_test.cpp
    multichannel_decay_test.cpp
    observables_test.cpp
    parallel_test.cpp
    phasor_test.cpp
    potential_test.cpp
    projection_test.cpp
    propagator_tables_test.cpp
    propagator_test.cpp
    radial_test.cpp
    regularized_coulomb_test.cpp
    relaxation_in_simulation_test.cpp
    rotation_test.cpp
    sampling_test.cpp
    simulation_state_test.cpp
    simulation_test.cpp
    solvable_wells_test.cpp
    spectral_test.cpp
    sphere_test.cpp
    time_scale_realtime_test.cpp
    trap_ladder_test.cpp
    volume_test.cpp
    vram_budget_test.cpp
    wavepacket3_test.cpp
    wavepacket_test.cpp
)
target_link_libraries(sesolver_tests
    PRIVATE
        sesolver_core
        ses_scenario
        GTest::gtest_main
)
target_compile_features(sesolver_tests PRIVATE cxx_std_20)

include(GoogleTest)
gtest_discover_tests(sesolver_tests)
