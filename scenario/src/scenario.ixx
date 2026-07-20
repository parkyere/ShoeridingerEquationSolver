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
    virtual double bfield_b() const = 0;  // director truth (panel syncs back)
    virtual void toggle_bfield_axis() = 0;
    virtual int bfield_axis() const = 0;
    virtual double ionized_fraction() const = 0;
    virtual void set_mcwf_damping(bool on) = 0;
    virtual bool mcwf_damping() const = 0;
    virtual double channel_a(int from, int to) const = 0;
    virtual double state_energy(int idx) const = 0;
    virtual long long photon_count() const = 0;
    // Spectrometer record: the energies (eV) of the photons emitted by
    // quantum jumps since the last reset, in emission order. spectro_max_ev
    // is the strip's full scale (the ionization limit) -- 0 hides the
    // strip (0 defaults: hydrogen is the sole HydrogenApi implementer;
    // emission rules are hydrogen-only).
    virtual int spectro_count() const { return 0; }
    virtual double spectro_ev(int /*i*/) const { return 0.0; }
    virtual double spectro_max_ev() const { return 0.0; }
    virtual int last_measured_index() const = 0;
    virtual void seed_kepler() = 0;  // circular-state Rydberg packet (K)
    virtual double mean_z() = 0;
    virtual double mean_x() = 0;  // Kepler-orbit readout pair
    virtual double mean_y() = 0;
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
    // Schrodinger-cat decoherence lens: the two-lobe cat |a> + |-a> and
    // the cavity photon-loss MCWF (each lost photon flips the parity).
    virtual void cat() = 0;
    virtual void toggle_loss() = 0;
    virtual bool loss_on() const = 0;
    virtual long long jump_count() const = 0;
    // The linear-combination spectrum strip (0..100 eV): what the cloud
    // IS made of. Weights only change on MUTATIONS (unitary evolution
    // leaves |c_n|^2 alone), so implementations recompute lazily.
    virtual int spectrum_count() = 0;
    virtual double spectrum_ev(int i) = 0;
    virtual double spectrum_weight(int i) = 0;
};

// The double-well tunneling-oscillation scene.
struct DoubleWellApi {
    virtual ~DoubleWellApi() = default;
    virtual double splitting() const = 0;  // dE = E1 - E0 (Ha)
    virtual double p_left() const = 0;
    virtual double p_right() const = 0;
    // Re-prepare psi_L in a well with the new barrier (a preparation demo).
    virtual void set_barrier(double vb) = 0;
    virtual double barrier() const = 0;
};

// The reflectionless (Poschl-Teller) scattering scene.
struct ReflectApi {
    virtual ~ReflectApi() = default;
    // Largest negative-momentum fraction seen while most of the norm was
    // still in the box (the honest R; absorbed flux must not inflate it).
    virtual double reflected_max() const = 0;
    virtual bool square_well() const = 0;
    virtual void toggle_well() = 0;  // sech^2 <-> equal square, relaunch
};

// One-electron fixed-nuclei molecules (Born-Oppenheimer): staged state
// preparation (ITP ground, then the deflated excited chain) + geometry
// control. prepare(k) is ASYNC (the relax runs over frames); the arcs and
// the panel poll prepared(k).
struct MoleculeApi {
    virtual ~MoleculeApi() = default;
    virtual bool prepared(int k) const = 0;  // state k solved and cached
    virtual double energy(int k) const = 0;  // captured E_k (Ha); 0 = none
    virtual void prepare(int k) = 0;         // relax the chain up to k
    virtual double nuclear_repulsion() const = 0;  // sum_{i<j} Z^2 / r_ij
    virtual void set_geometry(int variant) = 0;    // scene-defined presets
    virtual int geometry() const = 0;
    virtual void set_parameter(double p) = 0;  // scene knob (R / delta)
    virtual double parameter() const = 0;
    // P(|r| < radius): the real-time containment probe -- a prepared bound
    // state must STAY on the nuclei (arc contract for the step accuracy).
    virtual double containment(double radius) = 0;
};

// The Morse anharmonic-ladder scene (eigenstate jumps, shrinking gaps).
struct MorseApi {
    virtual ~MorseApi() = default;
    virtual int level() const = 0;           // -1 = pair superposition
    virtual double level_energy() const = 0;
    virtual bool jump(bool up) = 0;
    virtual int bound_count() const = 0;
};

// The 2D double-slit + Aharonov-Bohm scene: geometry sliders re-fire a
// fresh electron through the pierced wall; flux is the solenoid buried
// mid-wall between the slits (exact Peierls link phases, B = 0 on every
// electron path).
struct SlitApi {
    virtual ~SlitApi() = default;
    virtual void set_separation(double d) = 0;
    virtual double separation() const = 0;
    virtual void set_width(double w) = 0;
    virtual double width() const = 0;
    virtual void set_flux(double phi) = 0;
    virtual double flux() const = 0;
    virtual void refire() = 0;
    virtual double transmitted_fraction() const = 0;
    // Accumulated screen density at the row nearest y (arcs probe the
    // bright/dark fringes with this).
    virtual double screen_at(double y) const = 0;
};

// Landau / cyclotron scene: uniform B along z on the 2D lattice.
struct LandauApi {
    virtual ~LandauApi() = default;
    virtual void set_field(double b) = 0;
    virtual double field() const = 0;
    virtual void set_k0(double k0) = 0;
    virtual double k0() const = 0;
    virtual void refire() = 0;
    virtual double omega_c() const = 0;       // = B
    virtual double radius_pred() const = 0;   // k0 / B
    virtual double orbit_x() const = 0;       // live <x>
    virtual double orbit_y() const = 0;       // live <y>
    virtual double mean_n() const = 0;        // <E>/B - 1/2 (Landau index)
    // Recorded AT the crossings (step-chunk granularity; arcs poll far
    // too coarsely to catch an orbital phase): distance from the T/2
    // antipode and from the T = 2 pi / B start point. -1 until reached.
    virtual double antipode_dist() const = 0;
    virtual double closure_dist() const = 0;
    // Landau-level ladder (ses.lattice2d landau_ladder): a-dag / a jump one
    // cyclotron quantum; false = refused (a annihilated the lowest level).
    virtual bool ladder(bool up) = 0;
};

// 1D periodic-lattice (Bloch) scene: V0 sin^2(kL x) + tilt force F.
struct BlochApi {
    virtual ~BlochApi() = default;
    virtual void set_depth(double v0) = 0;
    virtual double depth() const = 0;
    virtual void set_force(double f) = 0;
    virtual double force() const = 0;
    virtual void refire() = 0;
    virtual double bloch_period() const = 0;  // G/F (0 = no tilt)
    virtual double quasimomentum() const = 0; // q(t) folded to the BZ
    virtual double mean_x() const = 0;        // live <x>
    virtual double excursion() const = 0;     // max |<x> - x0| since fire
};

// The 1993 IBM quantum corral: adatoms on a ring, standing-wave states
// relaxed inside the fence. States capture ASYNC over frames (poll
// relaxing()); energies in Hartree.
struct CorralApi {
    virtual ~CorralApi() = default;
    virtual void set_radius(double r) = 0;
    virtual double radius() const = 0;
    virtual double mass() const = 0;  // Cu(111) surface-state m* (a.u.)
    virtual void relax_next() = 0;  // capture the next (deflated) state
    virtual int captured() const = 0;
    virtual double energy(int k) const = 0;
    virtual double confinement() const = 0;  // probability inside the ring
    virtual bool relaxing() const = 0;
    virtual void fire_packet() = 0;  // scatter a packet off the fence
    // The state the STM images: the standing wave AT the Fermi energy
    // (k_F R ~ j0_10 => ~10 radial nodes), not the relaxed ground.
    virtual void fermi_wave() = 0;
};

// 2D quantum dot: parabolic confinement + optional uniform B -- the
// Fock-Darwin problem. Ground relax is ASYNC (poll relaxing()).
struct QdotApi {
    virtual ~QdotApi() = default;
    virtual void set_omega0(double w0) = 0;
    virtual double omega0() const = 0;
    virtual void set_field(double b) = 0;
    virtual double field() const = 0;
    virtual void relax_ground() = 0;
    virtual bool relaxing() const = 0;
    virtual double energy_meas() const = 0;
    virtual double energy_pred() const = 0;  // Omega = sqrt(w0^2 + B^2/4)
    virtual void fire_displaced() = 0;  // coherent orbit / rosette
    // 2D-HO extensions: circular ladder (B = 0 only -- gauge), a random
    // coherent packet, and the pick-and-gather grab (drag the surface;
    // time stands still while held, resumes on release).
    virtual bool ho_ladder(bool up) = 0;
    virtual void random_packet() = 0;
    virtual void begin_grab(double x, double y) = 0;
    virtual void update_grab(double strength) = 0;
    virtual void end_grab() = 0;
    virtual bool grabbing() const = 0;
    // Fock-Darwin linear-combination spectrum (0..100 eV), lazy like the
    // 1D trap's.
    virtual int spectrum_count() = 0;
    virtual double spectrum_ev(int i) = 0;
    virtual double spectrum_weight(int i) = 0;
};

// Quantum bouncer: gravity + mirror, the Airy ladder.
struct BouncerApi {
    virtual ~BouncerApi() = default;
    virtual void relax_ground() = 0;  // instant ITP anneal to Airy 1
    virtual void drop() = 0;          // packet from height: bounce/revive
    virtual double energy() const = 0;
    virtual double airy_e1() const = 0;  // ideal hard-floor E1
};

// Quantum point contact: conductance staircase in the gap width.
struct QpcApi {
    virtual ~QpcApi() = default;
    virtual void set_gap(double w) = 0;  // constriction width (refires)
    virtual double gap() const = 0;
    virtual void fire() = 0;
    virtual double transmitted() const = 0;  // right-cap flux tally
    virtual int open_channels() const = 0;   // floor(w k0 / pi)
};

// Quantum carpet: free ring, temporal Talbot weave.
struct CarpetApi {
    virtual ~CarpetApi() = default;
    virtual void refire() = 0;
    virtual double revival_time() const = 0;     // T_rev = L^2 / pi
    virtual double revival_overlap() const = 0;  // |<psi0|psi>|^2 live
    // Row-cadence maxima (frame polling would miss the ~2 au peak):
    virtual double mid_scramble_max() const = 0;  // max in (0.15, 0.6) T
    virtual double best_revival() const = 0;      // max past 0.6 T
};

// Anderson localization: 1D speckle wire, conductance framing.
struct AndersonApi {
    virtual ~AndersonApi() = default;
    virtual void set_disorder(double w) = 0;  // speckle strength (refires)
    virtual double disorder() const = 0;
    virtual void reroll() = 0;  // fresh landscape (new seed, refires)
    virtual void refire() = 0;
    virtual double transmitted() const = 0;  // right-cap flux tally
    virtual double survived() const = 0;     // on-stage norm
};

// Quantum billiard: circle (integrable) vs Bunimovich stadium (chaotic).
struct BilliardApi {
    virtual ~BilliardApi() = default;
    virtual void fire() = 0;          // relaunch the tangential packet
    virtual void toggle_shape() = 0;  // circle <-> stadium (refires)
    virtual bool stadium() const = 0;
    virtual void toggle_avg_view() = 0;  // live |psi|^2 <-> time average
    virtual bool avg_view() const = 0;
    virtual double avg_center_fraction() const = 0;  // caustic metric
};

// A 1D-scene overlay primitive: packed (x, y, z) float triples drawn in
// world space with a constant color -- a LINE_STRIP polyline, or with
// `fill` a TRIANGLE_STRIP sheet (the faint xy reference plane). When
// `rgba` is set (4 premultiplied floats per vertex) it REPLACES the
// constant color -- the phase-hued density band. Both pointers stay valid
// until the director's next run_frame().
struct OverlayCurve {
    const float* xyz = nullptr;
    int count = 0;
    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
    float a = 1.0f;
    bool fill = false;
    const float* rgba = nullptr;  // per-vertex premultiplied color
};

// A nucleus marker BALL (world space, symbolic radius): hydrogen's proton,
// the molecules' CPK atoms. Shaded as a real sphere in both views -- the
// mesh pipeline in Surface, the raymarcher in Cloud.
struct SceneMarker {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float radius = 0.35f;
    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
};

class ScenarioDirector {
public:
    virtual ~ScenarioDirector() = default;

    // Capability queries: non-null only for the scene that implements them.
    virtual HydrogenApi* hydrogen() { return nullptr; }
    virtual TunnelApi* tunnel() { return nullptr; }
    virtual Ladder1dApi* ladder1d() { return nullptr; }
    virtual DoubleWellApi* doublewell() { return nullptr; }
    virtual ReflectApi* reflect() { return nullptr; }
    virtual MorseApi* morse() { return nullptr; }
    virtual MoleculeApi* molecule() { return nullptr; }
    virtual SlitApi* slit() { return nullptr; }
    virtual LandauApi* landau() { return nullptr; }
    virtual BlochApi* bloch() { return nullptr; }
    virtual CorralApi* corral() { return nullptr; }
    virtual QdotApi* qdot() { return nullptr; }
    virtual BilliardApi* billiard() { return nullptr; }
    virtual AndersonApi* anderson() { return nullptr; }
    virtual CarpetApi* carpet() { return nullptr; }
    virtual QpcApi* qpc() { return nullptr; }
    virtual BouncerApi* bouncer() { return nullptr; }

    // 1D-scene overlay polylines (phasor curve + potential profile); the 3D
    // scenes return 0 and the renderer draws nothing extra.
    virtual int overlay_curve_count() const { return 0; }
    virtual OverlayCurve overlay_curve(int /*i*/) const { return {}; }

    // Photons emitted by quantum jumps, if the scene has any (generic so
    // arcs can probe every jump-capable scene; hydrogen's override serves
    // both this and HydrogenApi).
    virtual long long photon_count() const { return 0; }

    // Scene-prop hints for the renderer (display only, physics never reads
    // them): nucleus marker BALLS -- position, radius, color per ball (the
    // default single warm origin sphere serves hydrogen and the trap; the
    // molecules supply their CPK ball list; a barrier scene has NO point
    // to suggest and returns 0) -- and a visualized potential slab
    // [lo, hi) on x (the tunneling barrier).
    virtual int marker_count() const { return 1; }
    virtual SceneMarker marker(int /*i*/) const {
        return {0.0f, 0.0f, 0.0f, 0.35f, 1.0f, 0.45f, 0.20f};
    }
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
    // Real time = x1 pacing: every route back (key 1 / panel button) must
    // also clear the visualized time scale, or a raised slider survives as
    // a sticky speedup. NVI: scenes override do_set_real_time().
    void set_real_time() {
        do_set_real_time();
        set_time_scale(1);
    }
    virtual void do_set_real_time() = 0;
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
    // Integrator steps a x1 tick SUPPLIES: the readout's honest baseline is
    // ticks/s x dt x THIS (a scene feeding 8 steps/tick showed "x8.0" at
    // the x1 dial before this seam).
    virtual int steps_per_tick_x1() const { return 1; }

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
