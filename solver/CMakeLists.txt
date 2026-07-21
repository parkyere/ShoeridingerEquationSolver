# ses_solver: split-operator TDSE engine on framework-free Vulkan (headless).
find_package(VulkanHeaders CONFIG REQUIRED)  # vcpkg port -> Vulkan::Headers target
find_package(volk CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

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
        spin_site_gate spin_bond_gate spin_site_bloch spin_fused_gate
        fft_line8 fft_line64 fft_line256 fft_line512)
    ses_bake_shader(_ses_solver_hdrs "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
                    "${_ses_solver_spv_dir}" "${_k}.comp" compute "${_k}")
endforeach()
add_custom_target(ses_solver_spv DEPENDS ${_ses_solver_hdrs})

# The ONE VMA implementation TU for all of ses_vk.
add_library(ses_solver STATIC
    src/vma_impl.cpp
)
target_sources(ses_solver
    PUBLIC FILE_SET CXX_MODULES FILES
        src/vk_device.ixx
        src/vk_compute.ixx
        src/vk_engine.ixx
        src/engine_blobs.ixx
        src/spin_engine.ixx
        src/vram_probe.ixx)
target_compile_features(ses_solver PUBLIC cxx_std_20)
add_dependencies(ses_solver ses_solver_spv)
target_include_directories(ses_solver PRIVATE "${_ses_solver_spv_dir}")
# PUBLIC: solver BMIs import core BMIs; ses_arch codegen flags must match
# lib and consumers (BMI/importer mismatch = ODR trap).
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

# Headless GPU harness. SES_VK_VALIDATION=1 enables validation layers + fails on layer error.
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
    SKIP_RETURN_CODE 77)  # exit 77 = ctest SKIP when no Vulkan device present
message(STATUS "Framework-free harness 'sesolver_vkcheck' enabled.")
