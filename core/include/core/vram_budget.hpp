#pragma once

// Pick the resident eigenstate-atlas precision that fits GPU VRAM.
//
// The n<=6 tracked manifold keeps every m-resolved bound state resident on the
// GPU as a complex-fp32 buffer (256^3 -> 134,217,728 B each; 91 states ~=
// 12.2 GB). On a card that small (e.g. an 8 GB RTX 4060) the fp32 atlas does
// not fit: the WDDM driver pages the surplus into system RAM ("shared GPU
// memory"), host RAM balloons, and every frame thrashes across PCIe. The cure
// is to store the atlas at half precision (fp16, ~6.1 GB) when -- and only
// when -- the measured free VRAM cannot hold it at fp32.
//
// This is the PURE decision: given the free VRAM, the state count, and the
// per-state fp32 size, return which precision to use. It takes only integers
// and has zero Qt/OpenGL dependencies, so it obeys the sesolver_core reuse
// boundary and is unit-tested directly. The GL query that measures free VRAM
// (NVX_gpu_memory_info / ATI_meminfo) is the untested Humble-Object shell in
// app/, which passes kVramUnknown here when no such extension exists.

#include <cstdint>

namespace ses {

enum class GpuPrecision { Fp32, Fp16 };

// Sentinel the GL shell passes when free VRAM could not be measured (neither
// the NVIDIA nor the AMD memory-info extension is present).
inline constexpr std::int64_t kVramUnknown = -1;

// Choose the atlas state-buffer precision.
//   free_vram_bytes      : free VRAM at startup, or kVramUnknown if unmeasured.
//   num_states           : resident state buffers (91 for the n<=6 manifold).
//   bytes_per_state_fp32 : one complex-fp32 state (256^3 -> 134,217,728 B).
//   headroom_bytes       : VRAM the caller reserves for the volume texture,
//                          framebuffers, working buffers, and driver overhead.
//   out_fits (optional)  : set false when even the fp16 atlas overflows the
//                          budget, so the caller can warn; true otherwise.
// fp16 halves each complex scalar, so its footprint is bytes_per_state_fp32/2
// per state. An unmeasurable budget never silently degrades fidelity: it keeps
// fp32 (correct on the big-VRAM machines the app targets; the small-VRAM case
// is exactly the one the extensions do report).
inline GpuPrecision choose_state_precision(std::int64_t free_vram_bytes,
                                           int num_states,
                                           std::int64_t bytes_per_state_fp32,
                                           std::int64_t headroom_bytes,
                                           bool* out_fits = nullptr) {
    const auto set_fits = [&](bool v) {
        if (out_fits != nullptr) {
            *out_fits = v;
        }
    };
    if (free_vram_bytes == kVramUnknown || free_vram_bytes < 0) {
        set_fits(true);
        return GpuPrecision::Fp32;
    }
    const std::int64_t budget = free_vram_bytes - headroom_bytes;  // may be < 0
    const std::int64_t need32 =
        static_cast<std::int64_t>(num_states) * bytes_per_state_fp32;
    const std::int64_t need16 =
        static_cast<std::int64_t>(num_states) * (bytes_per_state_fp32 / 2);
    if (need32 <= budget) {
        set_fits(true);
        return GpuPrecision::Fp32;
    }
    if (need16 <= budget) {
        set_fits(true);
        return GpuPrecision::Fp16;
    }
    set_fits(false);  // even fp16 overflows: best effort, and warn.
    return GpuPrecision::Fp16;
}

}  // namespace ses
