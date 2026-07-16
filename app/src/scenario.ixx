module;
#include <volk.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
export module ses.app.scenario;
export import ses.vk.device;
export import ses.grid;
export import ses.marching_cubes;
export import ses.colormap;


// The scenario seam: everything a demo IS, behind one framework-neutral
// interface. The shell owns exactly one ScenarioDirector (chosen by --scene=)
// and talks to it through this contract; scenario-specific keys go through
// handle_key (plain ASCII -- the shell maps its own key codes down to this).
// ses.vk.device's GMF set, textually pre-claimed: volk.h both supplies the
// VK_* macros (macros never cross module boundaries) and inoculates the TU
// against GMF/textual std redefinitions.


export namespace ses_shell {

// Scenario-specific capability seams. The shell holds ONE ScenarioDirector
// and never down-casts to a concrete type; a scene that supports these
// controls returns a non-null pointer from the matching accessor, so the
// panel and the selftest arcs reach the specialized surface through a stable
// interface.
struct HydrogenApi {
    virtual ~HydrogenApi() = default;
    virtual void set_relaxing() = 0;
    virtual void relax_to_excited() = 0;
    virtual void relax_to_2s() = 0;
    virtual void excite_n3() = 0;
    virtual void toggle_decay() = 0;
    virtual void toggle_laser() = 0;
    virtual void measure_energy_now() = 0;
    virtual void measure_n_shell_now() = 0;
    virtual void measure_l_now() = 0;
    virtual void measure_m_now() = 0;
    virtual int last_partial_outcome() const = 0;
    virtual void set_efield_e0(double e0) = 0;
    virtual void set_bfield_b(double b) = 0;
    virtual void toggle_bfield_axis() = 0;
    virtual int bfield_axis() const = 0;
    virtual double ionized_fraction() const = 0;
    virtual void set_mcwf_damping(bool on) = 0;
    virtual bool mcwf_damping() const = 0;
    virtual double channel_a(int from, int to) const = 0;
    virtual double state_energy(int idx) const = 0;
    virtual long long photon_count() const = 0;
    virtual int last_measured_index() const = 0;
    virtual double mean_z() = 0;
    virtual double peak_excited_population() const = 0;
    virtual void debug_prepare_state(int idx) = 0;
    virtual double probe_population(int idx) = 0;
    virtual void debug_prepare_superposition(int a, int b) = 0;
};

struct TunnelApi {
    virtual ~TunnelApi() = default;
    virtual double transmitted_max() const = 0;
};

class ScenarioDirector {
public:
    virtual ~ScenarioDirector() = default;

    // Capability queries: non-null only for the scene that implements them.
    virtual HydrogenApi* hydrogen() { return nullptr; }
    virtual TunnelApi* tunnel() { return nullptr; }

    // ---- lifecycle ----
    virtual const ses::Grid3D& grid() const = 0;
    virtual void init_compute(ses_vk::DeviceContext& ctx, bool device_ok,
                              std::int64_t free_vram_bytes) = 0;
    virtual void release_gpu() = 0;
    virtual bool compute_attempted() const = 0;
    virtual bool gpu_ok() const = 0;

    // ---- per-frame / per-tick ----
    virtual void run_frame() = 0;
    virtual void tick() = 0;

    // ---- controls every scenario supports ----
    virtual void set_real_time() = 0;
    virtual void reset_simulation() = 0;
    virtual void measure_now() = 0;
    virtual void toggle_view_mode() = 0;
    // Scenario-specific keys (upper-case letters / digits); true = handled.
    virtual bool handle_key(char key) = 0;

    // ---- state the shell gates on ----
    virtual bool solving() const = 0;      // startup solve owns the GPU state
    virtual bool scene_ready() const = 0;  // demo fully armed (selftest gate)

    // Camera start distance framing this scene's box (Bohr).
    virtual double default_camera_distance() const { return 150.0; }

    // Visualized time scale: multiply the steps SUPPLIED per wall tick (and
    // the per-frame consumption cap). dt is untouched -- more integrator
    // steps per rendered frame, never larger ones -- so accuracy is
    // preserved and the GPU saturating just lowers fps honestly.
    virtual void set_time_scale(int scale) { (void)scale; }
    virtual int time_scale() const { return 1; }

    // Total simulated time (au) and the integrator step (au) -- the shell's
    // performance readout derives the achieved au/s from these.
    virtual double sim_time() const { return 0.0; }
    virtual double sim_dt() const { return 0.0; }

    // ---- display accessors (FrameInput assembly + title) ----
    virtual bool cloud() const = 0;
    virtual double peak() const = 0;
    virtual VkImageView psi_volume_view() = 0;
    // Low-res fp32 Bohmian-velocity volume for the streaklines (null -> the
    // renderer skips flow). Default null: only the GPU cloud scenes provide it.
    virtual VkImageView flow_velocity_view() { return VK_NULL_HANDLE; }
    // GPU surface mesh (non-null when the scene extracts on-GPU): the
    // renderer draws these directly and ignores mesh()/colors().
    virtual VkBuffer surface_vbuf() const { return VK_NULL_HANDLE; }
    virtual VkBuffer surface_indirect() const { return VK_NULL_HANDLE; }
    // GPU-surface vertex capacity: generous for any tracked isosurface
    // (transient while Surface is active); overflow clamps to a clean prefix
    // (the engine warns once).
    static constexpr int kMcMaxTris = 1000000;
    virtual float next_flash_intensity() = 0;
    virtual bool take_volume_written() = 0;
    virtual bool take_volume_dirty() = 0;
    virtual bool take_mesh_dirty() = 0;
    virtual void mark_display_dirty() = 0;
    virtual bool take_title_dirty() = 0;
    virtual const std::vector<float>& psi_staging() const = 0;
    virtual const ses::Mesh& mesh() const = 0;
    virtual const std::vector<ses::Rgb>& colors() const = 0;
    virtual std::string title_text() = 0;
};

}  // namespace ses_shell
