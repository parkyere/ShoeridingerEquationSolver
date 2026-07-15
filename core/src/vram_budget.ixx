module;
#include <cstdint>
export module ses.vram_budget;


// Pick the resident eigenstate-atlas precision (fp32 vs fp16) that fits GPU
// VRAM: an fp32 atlas that overflows VRAM makes WDDM page across PCIe and
// thrash every frame. Pure integer decision (no GPU-API deps, unit-tested);
// the VK_EXT_memory_budget query lives in the app/ shell, which passes
// kVramUnknown when unavailable.


export namespace ses {

enum class GpuPrecision { Fp32, Fp16 };

// Sentinel the app shell passes when free VRAM could not be measured
// (VK_EXT_memory_budget unavailable).
inline constexpr std::int64_t kVramUnknown = -1;

// Choose the atlas state-buffer precision. headroom_bytes = VRAM reserved for
// textures/framebuffers/driver overhead; out_fits is set false when even fp16
// overflows (caller warns). An unmeasurable budget never silently degrades
// fidelity: it keeps fp32.
inline constexpr GpuPrecision choose_state_precision(std::int64_t free_vram_bytes,
                                                     int num_states,
                                                     std::int64_t bytes_per_state_fp32,
                                                     std::int64_t headroom_bytes,
                                                     bool* out_fits = nullptr) noexcept {
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
