# ---------------------------------------------------------------------------
# ses_solver: the Schrodinger GPGPU library -- the split-operator TDSE engine
# (real time, imaginary-time relaxation, MCWF, projections, marching-cubes
# extraction) on framework-free Vulkan (volk + VMA + offline-baked Slang
# kernels). NO windowing, NO UI, NO renderer: sesolver_vkcheck (below) is the
# headless proof and the GPU verification harness. Consumers import
# ses.vk.device / ses.vk.compute / ses.vk.engine / ses.vk.engine_blobs.
# ---------------------------------------------------------------------------
find_package(VulkanHeaders CONFIG REQUIRED)  # vcpkg port: Vulkan::Headers
find_package(volk CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

# Engine compute kernels, baked to embedded SPIR-V.
set(_ses_solver_spv_dir "${CMAKE_CURRENT_BINARY_DIR}/spv")
file(MAKE_DIRECTORY "${_ses_solver_spv_dir}")
set(_ses_solver_hdrs "")
foreach(_k IN ITEMS phase_multiply scale norm_peak norm_finalize scale_buf
        inner_product
        dipole_kick mean_force dipole pack_half unpack_half conj_scale
        shear axpy copy_state synth project_deposit
        bridge_store bridge_load flow_velocity
        half_mul kin_mul damp_mul phase_damp_mul
        mcwf_axpy
        mc_density mc_classify mc_scan mc_emit
        spin_site_gate spin_bond_gate
        fft_line8 fft_line64 fft_line256 fft_line512)
    ses_bake_shader(_ses_solver_hdrs "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
                    "${_ses_solver_spv_dir}" "${_k}.comp" compute "${_k}")
endforeach()
add_custom_target(ses_solver_spv DEPENDS ${_ses_solver_hdrs})

# vma_impl.cpp is the ONE VMA implementation TU for the whole ses_vk world.
add_library(ses_solver STATIC
    src/vma_impl.cpp
)
target_sources(ses_solver
    PUBLIC FILE_SET CXX_MODULES FILES
        src/vk_device.ixx
        src/vk_compute.ixx
        src/vk_engine.ixx
        src/engine_blobs.ixx
        src/vram_probe.ixx)
target_compile_features(ses_solver PUBLIC cxx_std_20)
add_dependencies(ses_solver ses_solver_spv)
target_include_directories(ses_solver PRIVATE "${_ses_solver_spv_dir}")
# sesolver_core PUBLIC: solver BMIs import core BMIs, and ses_arch codegen
# flags must reach lib and consumers identically (BMI/importer mismatch =
# ODR trap).
target_link_libraries(ses_solver
    PUBLIC
        sesolver_core
        volk::volk
        GPUOpen::VulkanMemoryAllocator
        Vulkan::Headers
    PRIVATE
        ses_warnings)
ses_enable_vkfft(ses_solver)
if(MSVC)
    target_compile_definitions(ses_solver PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# ---------------------------------------------------------------------------
# sesolver_vkcheck: THE GPU verification harness. NO windowing anywhere in
# this binary (it builds and runs without SDL/ImGui installed). Every check
# pins a CPU double-precision reference value and tolerance. ctest: label
# "gpu", exit 77 = SKIP without a Vulkan device; SES_VK_VALIDATION=1 turns on
# the validation layers and fails the run on any layer error.
# ---------------------------------------------------------------------------
add_executable(sesolver_vkcheck src/vkcheck_main.cpp)
add_dependencies(sesolver_vkcheck ses_solver_spv)
if(MSVC)
    target_compile_definitions(sesolver_vkcheck PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
target_include_directories(sesolver_vkcheck PRIVATE "${_ses_solver_spv_dir}")
target_link_libraries(sesolver_vkcheck PRIVATE
    ses_solver
    volk::volk
    GPUOpen::VulkanMemoryAllocator
    ses_warnings)
target_compile_features(sesolver_vkcheck PRIVATE cxx_std_20)
ses_enable_vkfft(sesolver_vkcheck)
add_test(NAME VkCheck COMMAND sesolver_vkcheck)
set_tests_properties(VkCheck PROPERTIES
    LABELS "gpu"
    SKIP_RETURN_CODE 77)
message(STATUS "Framework-free harness 'sesolver_vkcheck' enabled.")
