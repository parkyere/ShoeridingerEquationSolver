module;
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
#include <norm_finalize_spv.h>
#include <scale_spv.h>
#include <scale_buf_spv.h>
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
#include <flow_velocity_spv.h>
#include <pack_half_spv.h>
#include <unpack_half_spv.h>
#include <fft_line8_spv.h>
#include <fft_line64_spv.h>
#include <fft_line256_spv.h>
#include <fft_line512_spv.h>
#include <cstdio>
export module ses.vk.engine_blobs;
export import ses.vk.engine;


// The solver library's embedded SPIR-V blobs: every compute kernel the
// ses_vk engine runs, baked offline from solver/shaders/ -- the engine has
// no resource system, so callers hand it this table. The line-FFT kernel is
// baked at concrete sizes; pick the one matching the transform axes (the
// cubic side, or the shared x/y side of an n x n x 1 planar grid).


export namespace ses_vk {

// Blobs for transform axes of length n; fft is null (engine init fails
// cleanly) if no fft_line kernel was baked at that size.
inline EngineKernels engine_blobs(int n) {
    EngineKernels b;
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
    b.norm_finalize = k_norm_finalize_spv;
    b.norm_finalize_size = k_norm_finalize_spv_size;
    b.scale_buf = k_scale_buf_spv;
    b.scale_buf_size = k_scale_buf_spv_size;
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
    b.flow_velocity = k_flow_velocity_spv;
    b.flow_velocity_size = k_flow_velocity_spv_size;
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
        case 512:
            b.fft = k_fft_line512_spv;
            b.fft_size = k_fft_line512_spv_size;
            break;
        default:
            std::fprintf(stderr, "engine_blobs: no fft_line%d kernel baked\n", n);
            break;
    }
    return b;
}

}  // namespace ses_vk
