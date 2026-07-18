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
export module ses.scenario;
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

// 1D harmonic-oscillator ladder controls (the textbook 1D trap scene).
struct Ladder1dApi {
    virtual ~Ladder1dApi() = default;
    // Current Fock level n; -1 = the state is a superposition (classified
    // honestly by Var(H), not bookkeeping).
    virtual int level() const = 0;
    virtual double level_energy() const = 0;  // live <H> (Ha)
    // Apply a-dag (up) / a (down); false = refused (a|0> = 0, or the
    // spectral-band cap on the way up).
    virtual bool ladder(bool up) = 0;
    // Well stiffness omega (width ~ 1/sqrt(w)). Setting it is a sudden
    // QUENCH: psi is kept and breathes in the new well; the reset target
    // becomes the new ground.
    virtual void set_omega(double w) = 0;
    virtual double omega() const = 0;
    // Largest ladder level reachable cleanly from the CURRENT state: an
    // eigenstate rungs via the stable oracle-rebuilt path (grid
    // representability ceiling, ses.ladder ho_level_cap); a superposition
    // rungs in the truncated Fock basis (ses.ladder ladder_fock band).
    virtual int max_level() const = 0;
    // Prepare a random coherent superposition over the low Fock levels (a
    // PURE state -- a density-matrix mixture is not representable in a
    // wavefunction solver).
    virtual void random_superposition() = 0;
};

// A 1D-scene overlay primitive: packed (x, y, z) float triples drawn in
// world space with a constant color -- a LINE_STRIP polyline, or with
// `fill` a TRIANGLE_STRIP sheet (the faint xy reference plane). The xyz
// pointer stays valid until the director's next run_frame().
struct OverlayCurve {
    const float* xyz = nullptr;
    int count = 0;
    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
    float a = 1.0f;
    bool fill = false;
};

class ScenarioDirector {
public:
    virtual ~ScenarioDirector() = default;

    // Capability queries: non-null only for the scene that implements them.
    virtual HydrogenApi* hydrogen() { return nullptr; }
    virtual TunnelApi* tunnel() { return nullptr; }
    virtual Ladder1dApi* ladder1d() { return nullptr; }

    // 1D-scene overlay polylines (phasor curve + potential profile); the 3D
    // scenes return 0 and the renderer draws nothing extra.
    virtual int overlay_curve_count() const { return 0; }
    virtual OverlayCurve overlay_curve(int /*i*/) const { return {}; }

    // Photons emitted by quantum jumps, if the scene has any (generic so
    // arcs can probe every jump-capable scene; hydrogen's override serves
    // both this and HydrogenApi).
    virtual long long photon_count() const { return 0; }

    // Scene-prop hints for the renderer (display only, physics never reads
    // them): the origin marker (hydrogen's proton, the trap's center -- a
    // barrier scene has NO point to suggest), and a visualized potential
    // slab [lo, hi) on x (the tunneling barrier).
    virtual bool center_marker() const { return true; }
    virtual bool barrier_slab(double& /*lo*/, double& /*hi*/) const {
        return false;
    }

    // Scene-chosen boot camera (the shell owns it afterwards). The generic
    // 3/4 view suits central scenes; the tunneling scene wants the slab
    // edge-on (packet left, wall a thin stripe, transmission right).
    virtual double default_camera_azimuth() const { return 0.6; }
    virtual double default_camera_elevation() const { return 0.4; }

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
