#pragma once

// The app's embedded SPIR-V blobs: every compute kernel and render shader
// the ses_vk engine/renderer runs, baked offline by glslangValidator from
// the Vulkan-GLSL sources in src/shaders/ -- the engine has no resource
// system, so main() hands it this table. The line-FFT kernel is baked at
// concrete sizes; pick the one matching the (cubic) grid.

#include "vk_engine.hpp"
#include "vk_render.hpp"

#include <phase_multiply_spv.h>
#include <half_mul_spv.h>
#include <kin_mul_spv.h>
#include <damp_mul_spv.h>
#include <phase_damp_mul_spv.h>
#include <mcwf_axpy_spv.h>
#include <mc_density_spv.h>
#include <mc_classify_spv.h>
#include <mc_scan_spv.h>
#include <mc_emit_spv.h>
#include <conj_scale_spv.h>
#include <norm_peak_spv.h>
#include <scale_spv.h>
#include <dipole_kick_spv.h>
#include <shear_spv.h>
#include <inner_product_spv.h>
#include <axpy_spv.h>
#include <copy_state_spv.h>
#include <synth_spv.h>
#include <mean_force_spv.h>
#include <dipole_spv.h>
#include <project_deposit_spv.h>
#include <bridge_store_spv.h>
#include <bridge_load_spv.h>
#include <pack_half_spv.h>
#include <unpack_half_spv.h>
#include <fft_line8_spv.h>
#include <fft_line64_spv.h>
#include <fft_line256_spv.h>
#include <mesh_vert_spv.h>
#include <mesh_frag_spv.h>
#include <volume_vert_spv.h>
#include <volume_frag_spv.h>
#include <accum_spv.h>
#include <bloom_down_spv.h>
#include <bloom_up_spv.h>
#include <compose_spv.h>
#include <particles_spv.h>
#include <occupancy_spv.h>
#include <occ_dilate_spv.h>
#include <shadow_spv.h>
#include <flow_vert_spv.h>
#include <flow_frag_spv.h>

#include <cstdio>

namespace ses_shell {

// Blobs for a cubic grid of side n; fft is null (engine init fails cleanly)
// if no fft_line kernel was baked at that size.
inline ses_vk::EngineKernels app_engine_blobs(int n) {
    ses_vk::EngineKernels b;
    b.mul = k_phase_multiply_spv;
    b.mul_size = k_phase_multiply_spv_size;
    b.half_mul = k_half_mul_spv;
    b.half_mul_size = k_half_mul_spv_size;
    b.kin_mul = k_kin_mul_spv;
    b.kin_mul_size = k_kin_mul_spv_size;
    b.damp = k_damp_mul_spv;
    b.damp_size = k_damp_mul_spv_size;
    b.pd = k_phase_damp_mul_spv;
    b.pd_size = k_phase_damp_mul_spv_size;
    b.mcwf = k_mcwf_axpy_spv;
    b.mcwf_size = k_mcwf_axpy_spv_size;
    b.mc_density = k_mc_density_spv;
    b.mc_density_size = k_mc_density_spv_size;
    b.mc_classify = k_mc_classify_spv;
    b.mc_classify_size = k_mc_classify_spv_size;
    b.mc_scan = k_mc_scan_spv;
    b.mc_scan_size = k_mc_scan_spv_size;
    b.mc_emit = k_mc_emit_spv;
    b.mc_emit_size = k_mc_emit_spv_size;
    b.conj = k_conj_scale_spv;
    b.conj_size = k_conj_scale_spv_size;
    b.norm = k_norm_peak_spv;
    b.norm_size = k_norm_peak_spv_size;
    b.scale = k_scale_spv;
    b.scale_size = k_scale_spv_size;
    b.kick = k_dipole_kick_spv;
    b.kick_size = k_dipole_kick_spv_size;
    b.shear = k_shear_spv;
    b.shear_size = k_shear_spv_size;
    b.inner = k_inner_product_spv;
    b.inner_size = k_inner_product_spv_size;
    b.axpy = k_axpy_spv;
    b.axpy_size = k_axpy_spv_size;
    b.copy = k_copy_state_spv;
    b.copy_size = k_copy_state_spv_size;
    b.synth = k_synth_spv;
    b.synth_size = k_synth_spv_size;
    b.force = k_mean_force_spv;
    b.force_size = k_mean_force_spv_size;
    b.dipole = k_dipole_spv;
    b.dipole_size = k_dipole_spv_size;
    b.project = k_project_deposit_spv;
    b.project_size = k_project_deposit_spv_size;
    b.bridge_store = k_bridge_store_spv;
    b.bridge_store_size = k_bridge_store_spv_size;
    b.bridge_load = k_bridge_load_spv;
    b.bridge_load_size = k_bridge_load_spv_size;
    b.pack = k_pack_half_spv;
    b.pack_size = k_pack_half_spv_size;
    b.unpack = k_unpack_half_spv;
    b.unpack_size = k_unpack_half_spv_size;
    switch (n) {
        case 8:
            b.fft = k_fft_line8_spv;
            b.fft_size = k_fft_line8_spv_size;
            break;
        case 64:
            b.fft = k_fft_line64_spv;
            b.fft_size = k_fft_line64_spv_size;
            break;
        case 256:
            b.fft = k_fft_line256_spv;
            b.fft_size = k_fft_line256_spv_size;
            break;
        default:
            std::fprintf(stderr, "vk_blobs: no fft_line%d kernel baked\n", n);
            break;
    }
    return b;
}

// The raw-Vulkan scene renderer's shader pair set.
inline ses_vk::RenderKernels app_render_blobs() {
    ses_vk::RenderKernels r;
    r.mesh_vert = k_mesh_vert_spv;
    r.mesh_vert_size = k_mesh_vert_spv_size;
    r.mesh_frag = k_mesh_frag_spv;
    r.mesh_frag_size = k_mesh_frag_spv_size;
    r.volume_vert = k_volume_vert_spv;
    r.volume_vert_size = k_volume_vert_spv_size;
    r.volume_frag = k_volume_frag_spv;
    r.volume_frag_size = k_volume_frag_spv_size;
    r.accum = k_accum_spv;
    r.accum_size = k_accum_spv_size;
    r.bloom_down = k_bloom_down_spv;
    r.bloom_down_size = k_bloom_down_spv_size;
    r.bloom_up = k_bloom_up_spv;
    r.bloom_up_size = k_bloom_up_spv_size;
    r.compose = k_compose_spv;
    r.compose_size = k_compose_spv_size;
    r.particles = k_particles_spv;
    r.particles_size = k_particles_spv_size;
    r.occupancy = k_occupancy_spv;
    r.occupancy_size = k_occupancy_spv_size;
    r.occ_dilate = k_occ_dilate_spv;
    r.occ_dilate_size = k_occ_dilate_spv_size;
    r.shadow = k_shadow_spv;
    r.shadow_size = k_shadow_spv_size;
    r.flow_vert = k_flow_vert_spv;
    r.flow_vert_size = k_flow_vert_spv_size;
    r.flow_frag = k_flow_frag_spv;
    r.flow_frag_size = k_flow_frag_spv_size;
    return r;
}

}  // namespace ses_shell
