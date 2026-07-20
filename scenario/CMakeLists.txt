# ---------------------------------------------------------------------------
# ses_scenario: the physics-orchestration library -- everything a demo IS.
# ScenarioDirector (the framework-neutral seam), the tracked-manifold spec,
# the atom model, and the concrete hydrogen / harmonic / tunneling directors,
# plus the headless selftest arcs. Binds sesolver_core physics to the
# ses_solver GPU engine; knows NOTHING about SDL, ImGui, or the renderer
# (display accessors hand out opaque Vulkan handles and CPU meshes).
# Consumers import ses.scenario and ses.scenario.*.
# ---------------------------------------------------------------------------
add_library(ses_scenario STATIC)
target_sources(ses_scenario
    PUBLIC FILE_SET CXX_MODULES FILES
        src/scenario.ixx
        src/manifold_spec.ixx
        src/kepler_seed.ixx
        src/atom_model.ixx
        src/base_director.ixx
        src/hydrogen_director.ixx
        src/harmonic_director.ixx
        src/tunneling_director.ixx
        src/line1d_director.ixx
        src/harmonic1d_director.ixx
        src/tunneling1d_director.ixx
        src/doublewell1d_director.ixx
        src/ptwell1d_director.ixx
        src/morse1d_director.ixx
        src/lattice2d_director.ixx
        src/billiard2d_director.ixx
        src/anderson1d_director.ixx
        src/carpet1d_director.ixx
        src/qpc2d_director.ixx
        src/bouncer1d_director.ixx
        src/spin_director.ixx
        src/spins_director.ixx
        src/doubleslit2d_director.ixx
        src/landau2d_director.ixx
        src/bloch1d_director.ixx
        src/corral2d_director.ixx
        src/qdot2d_director.ixx
        src/molecule_director.ixx
        src/selftest_arcs.ixx)
target_compile_features(ses_scenario PUBLIC cxx_std_20)
target_link_libraries(ses_scenario
    PUBLIC
        ses_solver
    PRIVATE
        ses_warnings)
if(MSVC)
    target_compile_definitions(ses_scenario PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
