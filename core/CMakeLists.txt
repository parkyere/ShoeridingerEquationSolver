# sesolver_core: the pure numerical/physics core.
# HARD RULE: this target must NEVER depend on Qt, OpenGL/Vulkan, or any
# windowing / rendering library. Everything here is reachable by unit tests
# with analytic oracles (TDD loop: docs/TDD_RULES.md).
# Sources are the CXX_MODULES file set below -- each module interface unit
# compiles to a real object, so the archive needs no placeholder TU.
add_library(sesolver_core STATIC)

# C++20 modules the core provides (consumers `import` these) -- the whole
# core is modules now; the old core/include header tree is gone. PUBLIC so
# the app/test targets that link the core can import across the target
# boundary.
target_sources(sesolver_core
    PUBLIC FILE_SET CXX_MODULES FILES
        src/marching_cubes_tables.ixx
        src/colormap.ixx
        src/vram_budget.ixx
        src/flow.ixx
        src/cross_section.ixx
        src/camera.ixx
        src/volume.ixx
        src/emission.ixx
        src/vec.ixx
        src/grid.ixx
        src/spectral.ixx
        src/potential.ixx
        src/wavepacket.ixx
        src/sphere.ixx
        src/harmonics.ixx
        src/parallel.ixx
        src/field.ixx
        src/marching_cubes.ixx
        src/fft.ixx
        src/decay.ixx
        src/measurement.ixx
        src/radial.ixx
        src/observables.ixx
        src/bloch.ixx
        src/ladder.ixx
        src/lattice2d.ixx
        src/heightfield.ixx
        src/phasor.ixx
        src/spectrum1d.ixx
        src/rotation.ixx
        src/propagator.ixx
        src/imaginary_time.ixx
        src/sampling.ixx
        src/projection.ixx
        src/drive.ixx
        src/magnetic.ixx
        src/simulation.ixx)
target_compile_features(sesolver_core PUBLIC cxx_std_20)
target_link_libraries(sesolver_core PRIVATE ses_warnings)
target_link_libraries(sesolver_core PUBLIC ses_arch)

# CPU hot loops thread through ses.parallel (core/src/parallel.ixx), the
# project's own std::jthread pool: OpenMP is gone because MSVC miscompiles
# omp pragmas inside C++20 module interfaces (silent zero-output) and the
# pool's fixed-chunk reductions are bitwise-reproducible across machines.
find_package(Threads REQUIRED)
target_link_libraries(sesolver_core PUBLIC Threads::Threads)

# _CRT_SECURE_NO_WARNINGS keeps MSVC quiet about std::getenv (the
# SES_PARALLEL_WORKERS debug override in parallel.ixx); same idiom as the app.
if(MSVC)
    target_compile_definitions(sesolver_core PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
