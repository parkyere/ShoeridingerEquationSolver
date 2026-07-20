# Manual micro-benchmark target (not registered with ctest; timing assertions
# are flaky by nature -- correctness lives in tests/).
add_executable(sesolver_bench
    bench_main.cpp
)
target_link_libraries(sesolver_bench PRIVATE sesolver_core ses_warnings)
target_compile_features(sesolver_bench PRIVATE cxx_std_20)

# Offline code generator: bakes the exact prolate-spheroidal H2+ atlas into
# core/src/h2plus_atlas_data.ixx. Run manually to regenerate; not a test.
add_executable(sesolver_genatlas
    gen_h2plus_atlas.cpp
)
target_link_libraries(sesolver_genatlas PRIVATE sesolver_core ses_warnings)
target_compile_features(sesolver_genatlas PRIVATE cxx_std_20)
