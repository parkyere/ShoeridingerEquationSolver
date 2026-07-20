module;
#include <utility>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>
export module ses.scenario.molecule_director;
export import ses.scenario.base_director;
import ses.wavepacket;
import ses.scenario.molecular_seed;
import ses.spheroidal;
import ses.h2plus_atlas_data;


// One-electron molecules with FIXED nuclei (Born-Oppenheimer): the engine
// propagates the single electron in a multi-center Coulomb landscape; the
// nuclei are geometry, not dynamics. Two scenes share the machinery:
//
//  - H2+ (two bare protons, regularized cells): sigma_g bonding + deflated
//    sigma_u (nodal plane); the R knob scans E_total(R) = E_elec + 1/R --
//    the chemical bond is its minimum.
//  - Stripped benzene: the FIRST electron of C6H6^41+ over the BARE nuclei
//    (Z_C = 6, Z_H = 1, regularized cells, centers lattice-snapped). No
//    soft cores, no free parameters, REAL (uniform, X-ray) geometry only
//    -- project rules: bare regularized Coulomb everywhere, no
//    counterfactual knobs. Low spectrum: deep quasi-degenerate carbon-core
//    band (1s(Z=6) inter-carbon hopping ~e^{-16}: core orbitals, not a
//    delocalized pi system). CPK-convention markers: carbon-black and
//    hydrogen-white atom discs over a gray bond skeleton.
//
// State preparation is a CHAIN: ITP ground, then deflated excited states
// against the captured lower ones (fp32 state buffers, engine-resident),
// mirroring the hydrogen director's deflation flow. prepare(k) is async;
// completion is the ITP energy plateau. License physics (32^3 CPU):
// tests/molecule_test.cpp.


export namespace ses_shell {

class MoleculeDirectorBase : public BaseDirector, public MoleculeApi {
public:
    explicit MoleculeDirectorBase(ses::WavepacketSimulation sim)
        : BaseDirector(std::move(sim)) {}

    MoleculeApi* molecule() override { return this; }

    // ---- MoleculeApi ----
    bool prepared(int k) const override {
        return k >= 0 && k < kStates && prepared_[k];
    }
    double energy(int k) const override {
        return (k >= 0 && k < kStates) ? e_[k] : 0.0;
    }
    void prepare(int k) override {
        if (!gpu_ok_ || k < 0 || k >= kStates) {
            return;  // the deflation chain runs on the GPU path only
        }
        showing_random_ = false;
        want_ = k;
        advance_chain();
    }
    // P(|r| < radius) on the CPU truth (bridges the GPU state first).
    double containment(double radius) override {
        ensure_cpu_current();
        const ses::Grid3D& g = sim_.grid();
        const double r2 = radius * radius;
        double inside = 0.0;
        double total = 0.0;
        for (int k = 0; k < g.z.n; ++k) {
            const double z = g.z.coord(k);
            for (int j = 0; j < g.y.n; ++j) {
                const double y = g.y.coord(j);
                for (int i = 0; i < g.x.n; ++i) {
                    const double x = g.x.coord(i);
                    const double w = std::norm(sim_.psi()(i, j, k));
                    total += w;
                    if (x * x + y * y + z * z < r2) {
                        inside += w;
                    }
                }
            }
        }
        return total > 0.0 ? inside / total : 0.0;
    }
    int state_count() const override { return exposed_states(); }
    const char* orbital_label(int k) const override { return orbital_name(k); }
    // Cancel any relax chain, drop an arbitrary normalized state, evolve it.
    void seed_random() override {
        if (!gpu_ok_) {
            return;  // the scene lives on the GPU real-time path
        }
        fine_polish_ = false;
        fine_left_ = 0;
        engine_.release_relax_tables();
        sim_.set_psi(ses_shell::random_molecular_seed(sim_.grid(),
                                                      rand_seed_++));
        cpu_is_truth_ = true;  // run_frame uploads the seed to the engine
        stepping_ = BaseStepping::RealTime;
        showing_random_ = true;
        title_dirty_ = true;
        stage_active_view();
    }

    double nuclear_repulsion() const override {
        const std::vector<ses::Vec3d> c = centers();
        const std::vector<double> q = charges();
        double acc = 0.0;
        for (std::size_t i = 0; i < c.size(); ++i) {
            for (std::size_t j = i + 1; j < c.size(); ++j) {
                const double dx = c[i].x - c[j].x;
                const double dy = c[i].y - c[j].y;
                const double dz = c[i].z - c[j].z;
                acc += q[i] * q[j] / std::sqrt(dx * dx + dy * dy + dz * dz);
            }
        }
        return acc;
    }
    int geometry() const override { return geom_; }
    double parameter() const override { return param_; }
    void set_geometry(int variant) override {
        geom_ = variant;
        param_ = geometry_parameter(variant);
        apply_geometry();
    }
    void set_parameter(double p) override {
        // Drag memo (the hydrogen uploaded_* rule): ImGui fires every drag
        // frame; an unchanged SNAPPED geometry must not rebuild the 256^3
        // sim and throw the relax chain away.
        const double snapped = clamp_parameter(p);
        if (geom_ == -1 && snapped == param_) {
            return;
        }
        geom_ = -1;  // custom knob value
        param_ = snapped;
        apply_geometry();
    }

    bool handle_key(char key) override {
        if (key == 'S' || key == 's') {
            seed_random();
            return true;
        }
        // Keys 2 .. 2+state_count()-1 prepare the known orbitals in order.
        if (key >= '2' && key < static_cast<char>('2' + state_count())) {
            prepare(key - '2');
            return true;
        }
        return false;
    }

    // CPK nucleus balls, filled by the scenes' rebuild_markers(): real
    // shaded spheres in both views (the hydrogen scene's proton machinery,
    // generalized to a list).
    int marker_count() const override {
        return static_cast<int>(balls_.size());
    }
    SceneMarker marker(int i) const override {
        return balls_[static_cast<std::size_t>(i)];
    }

protected:
    static constexpr int kStates = 6;  // buffer capacity; scenes expose <= this
    // Fine-polish burst after the coarse plateau (see run_relax_batch).
    static constexpr double kMolFineDtau = 0.002;
    static constexpr int kMolFinePolishSteps = 600;  // tau ~ 1.2

    // ---- scene hooks ----
    // How many known orbitals the scene exposes, and each one's label.
    virtual int exposed_states() const = 0;
    virtual const char* orbital_name(int /*k*/) const { return ""; }
    virtual std::vector<ses::Vec3d> centers() const = 0;
    // Per-center effective charges (parallel to centers()); default all 1.
    virtual std::vector<double> charges() const {
        return std::vector<double>(centers().size(), 1.0);
    }
    // Deflation-chain seed for state k (benzene). H2+ overrides prepare()
    // to synthesize from its analytic atlas instead, so it needs none.
    virtual ses::Field3D excited_seed(int /*k*/) const { return ground_seed(); }
    virtual double geometry_parameter(int variant) const = 0;
    virtual double clamp_parameter(double p) const = 0;
    virtual void geometry_changed() {}  // rebuild markers etc.

    // Swap the nuclear geometry under a fresh electron: new potential on
    // both the CPU truth and the engine, stale state caches dropped, and
    // the ground relax restarted automatically (an E(R)-scan step).
    void apply_geometry() {
        if (gpu_ok_) {
            engine_.wait_async();
        }
        sim_ = remake_simulation();
        free_state_buffers();
        cpu_is_truth_ = true;
        gpu_time_ = 0.0;
        pending_gpu_steps_ = 0;
        stepping_ = BaseStepping::RealTime;
        if (gpu_ok_) {
            engine_.release_relax_tables();  // tables bake the OLD potential
            engine_.set_potential(sim_.potential());
            engine_.set_potential_gradient(sim_.potential());
        }
        geometry_changed();
        title_dirty_ = true;
        if (gpu_ok_) {
            prepare(0);
        } else {
            stage_active_view();
        }
    }

    // Boot straight into the ground state: an un-relaxed (non-eigenstate)
    // Gaussian seed wraps the periodic (absorber-free) box as ripples and
    // its core phase-rotates at |E| dt ~ rad/step; auto-relax opens the
    // scene on the stationary physics.
    void on_gpu_ready() override { prepare(0); }

    void advance_chain() {
        int next = -1;
        for (int k = 0; k <= want_; ++k) {
            if (!prepared_[k]) {
                next = k;
                break;
            }
        }
        if (next < 0) {
            return;  // everything up to want_ is already cached
        }
        if (next == target_ && stepping_ == BaseStepping::Relaxing) {
            return;  // that state's relax is already in flight
        }
        start_relax_for(next);
    }

    void start_relax_for(int k) {
        if (fine_polish_ || fine_left_ > 0) {
            fine_polish_ = false;  // a fresh relax always starts coarse
            fine_left_ = 0;
            engine_.release_relax_tables();
        }
        if (!ensure_relax_tables()) {
            return;
        }
        if (mode_ != BaseViewMode::Cloud) {
            mode_ = BaseViewMode::Cloud;
        }
        deflate_.clear();
        for (int i = 0; i < k; ++i) {
            deflate_.push_back(buf_[i]);
        }
        sim_.set_psi(k == 0 ? ground_seed() : excited_seed(k));
        cpu_is_truth_ = true;  // run_frame uploads the seed to the engine
        target_ = k;
        relax_plateau_ = 0;
        relax_prev_energy_ = 0.0;
        stepping_ = BaseStepping::Relaxing;
    }

    // Deflated ITP batch + plateau auto-complete + fine polish + capture.
    // The coarse tables (kBaseRelaxDtau) settle fast but their fixed point
    // is Trotter-biased in a deep well (benzene: V*dtau ~ 7.6 left ~60%
    // continuum junk that real time honestly dispersed over the box); a
    // FIXED fine-dtau burst then purges the junk before capture -- the
    // hydrogen post-collapse flush pattern at molecule scale.
    void run_relax_batch() override {
        const ses_vk::Engine::RelaxStats stats =
            deflate_.empty()
                ? engine_.relax_step(pending_gpu_steps_)
                : engine_.relax_deflated_step(deflate_, pending_gpu_steps_);
        relax_energy_display_ = stats.energy;
        if (stats.peak > 0.0) {
            peak_ = stats.peak;
        }
        norm_display_ = 1.0;
        if (fine_left_ > 0) {
            fine_left_ -= pending_gpu_steps_;  // fixed budget, no plateau
            if (fine_left_ <= 0) {
                complete_relax();
            }
            return;
        }
        if (gpu_title_due_) {
            if (std::abs(stats.energy - relax_prev_energy_) < 5e-5) {
                ++relax_plateau_;
            } else {
                relax_plateau_ = 0;
            }
            relax_prev_energy_ = stats.energy;
            if (relax_plateau_ >= 12) {
                relax_plateau_ = 0;
                enter_fine_polish();
            }
        }
    }

    void enter_fine_polish() {
        engine_.release_relax_tables();
        fine_polish_ = true;  // relax_dtau() flips to the fine step
        if (!ensure_relax_tables()) {
            fine_polish_ = false;
            complete_relax();  // no tables: capture the coarse state as-is
            return;
        }
        fine_left_ = kMolFinePolishSteps;
    }

    // Fine phase: V*dtau ~ 0.3 even at the benzene cell depth; the burst
    // spans tau ~ 1.2, so junk at dE >~ 10 Ha dies as e^{-12}.
    double relax_dtau() const override {
        return fine_polish_ ? kMolFineDtau : kBaseRelaxDtau;
    }

    void complete_relax() {
        fine_polish_ = false;
        fine_left_ = 0;
        engine_.release_relax_tables();  // next chain state rebuilds coarse
        stepping_ = BaseStepping::RealTime;
        // Capture the converged state as an engine-resident deflation
        // buffer + record its energy for the HUD/arcs.
        if (engine_.readback(readback_buf_)) {
            ses::Field3D f{sim_.grid()};
            for (std::size_t i = 0; i < f.data().size(); ++i) {
                f.data()[i] = std::complex<double>{readback_buf_[2 * i],
                                                   readback_buf_[2 * i + 1]};
            }
            ses::normalize(f);
            if (buf_[target_] >= 0) {
                engine_.release_state(buf_[target_]);
            }
            buf_[target_] = engine_.create_state_buffer(f.data());
            if (buf_[target_] >= 0) {
                e_[target_] = relax_energy_display_;
                prepared_[target_] = true;
            }
        }
        title_dirty_ = true;
        if (target_ < want_ && prepared_[target_]) {
            advance_chain();  // keep climbing the requested chain
        } else {
            engine_.release_relax_tables();
        }
    }

    void free_state_buffers() {
        for (int k = 0; k < kStates; ++k) {
            if (buf_[k] >= 0 && gpu_ok_) {
                engine_.release_state(buf_[k]);
            }
            buf_[k] = -1;
            prepared_[k] = false;
            e_[k] = 0.0;
        }
        deflate_.clear();
        want_ = 0;
        target_ = 0;
    }

    void after_reset() override { free_state_buffers(); }

    ses::Field3D ground_seed() const {
        return ses::gaussian_wavepacket(sim_.grid(), ses::Vec3d{},
                                        ses::Vec3d{1.8, 1.8, 1.8},
                                        ses::Vec3d{});
    }

    // The scenes' CPK ball helper.
    static SceneMarker ball(const ses::Vec3d& c, double radius, float r,
                            float g, float b) {
        return {static_cast<float>(c.x),
                static_cast<float>(c.y),
                static_cast<float>(c.z),
                static_cast<float>(radius),
                r,
                g,
                b};
    }

    std::vector<SceneMarker> balls_;
    int geom_ = 0;
    double param_ = 0.0;
    int buf_[kStates] = {-1, -1, -1, -1, -1, -1};
    double e_[kStates] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    bool prepared_[kStates] = {false, false, false, false, false, false};
    int want_ = 0;
    int target_ = 0;
    bool fine_polish_ = false;  // relax anneal stage (coarse -> fine)
    int fine_left_ = 0;         // remaining fine-burst steps (0 = coarse)
    bool showing_random_ = false;  // last drop was a random seed, not an MO
    std::uint64_t rand_seed_ = 0;  // random-seed counter (deterministic)
    std::vector<int> deflate_;
};

// ---- H2+ ------------------------------------------------------------------

constexpr double kH2pBox = 20.0;   // Bohr half-extent, 256^3 (h ~ 0.156)
constexpr int kH2pPoints = 256;
constexpr double kH2pDt = 0.04;
constexpr double kH2pRDefault = 2.0;  // near the true equilibrium R ~ 2.0
constexpr double kH2pRMin = 1.0;
constexpr double kH2pRMax = 8.0;

class H2PlusDirector final : public MoleculeDirectorBase {
public:
    H2PlusDirector() : MoleculeDirectorBase(make(kH2pRDefault)) {
        param_ = snap_r(kH2pRDefault);
        rebuild_markers();
    }

    double default_camera_azimuth() const override { return 0.35; }
    double default_camera_elevation() const override { return 0.28; }
    double default_camera_distance() const override { return 55.0; }

protected:
    const char* scene_name() const override { return "H2+ molecular ion"; }

    ses::WavepacketSimulation remake_simulation() const override {
        return make(param_ > 0.0 ? param_ : kH2pRDefault);
    }

    std::vector<ses::Vec3d> centers() const override {
        const double d = 0.5 * snap_r(param_);
        return {{-d, 0.0, 0.0}, {d, 0.0, 0.0}};
    }

    // ---- the analytic (prolate-spheroidal) orbital atlas ----------------
    // H2+ is EXACTLY separable, so the known orbitals are synthesized
    // directly from ses.spheroidal (no relaxation) -- the same
    // reduce-to-1D-and-synthesize atlas the hydrogen atom uses. The atlas
    // climbs to the representability ceiling (bound orbitals whose radial
    // extent fits the box). prepare(k) = instant synthesis of atlas MO k.

    int exposed_states() const override {
        ensure_atlas();
        return static_cast<int>(atlas_.size());
    }
    const char* orbital_name(int k) const override {
        ensure_atlas();
        return (k >= 0 && k < static_cast<int>(atlas_.size()))
                   ? atlas_[static_cast<std::size_t>(k)].label.c_str()
                   : "";
    }
    bool prepared(int k) const override {
        ensure_atlas();
        return k >= 0 && k < static_cast<int>(atlas_.size());
    }
    double energy(int k) const override {
        ensure_atlas();
        return (k >= 0 && k < static_cast<int>(atlas_.size()))
                   ? atlas_[static_cast<std::size_t>(k)].orb.energy
                   : 0.0;
    }
    void prepare(int k) override {
        ensure_atlas();
        if (k < 0 || k >= static_cast<int>(atlas_.size())) {
            return;
        }
        showing_random_ = false;
        cur_ = k;
        const Mo& mo = atlas_[static_cast<std::size_t>(k)];
        set_state_field(ses::synthesize_h2plus(sim_.grid(), mo.orb, mo.partner));
    }
    // Random = random NORMALIZED superposition of the atlas orbitals (a
    // legitimate bound state of arbitrary shape -- it stays on the molecule
    // and beats between MOs, unlike a raw random blob field).
    void seed_random() override {
        ensure_atlas();
        if (atlas_.empty()) {
            return;
        }
        ses::Field3D acc{sim_.grid()};
        std::mt19937_64 rng{rand_seed_++};
        std::normal_distribution<double> gauss(0.0, 1.0);
        for (const Mo& mo : atlas_) {
            const ses::Field3D o =
                ses::synthesize_h2plus(sim_.grid(), mo.orb, mo.partner);
            const std::complex<double> c{gauss(rng), gauss(rng)};
            for (int i = 0; i < acc.size(); ++i) {
                acc.data()[static_cast<std::size_t>(i)] +=
                    c * o.data()[static_cast<std::size_t>(i)];
            }
        }
        ses::normalize(acc);
        showing_random_ = true;
        set_state_field(acc);
    }

    void on_gpu_ready() override {
        ensure_atlas();
        prepare(0);
    }

    double geometry_parameter(int variant) const override {
        return variant == 1 ? 6.0 : kH2pRDefault;  // 0 = equilibrium, 1 = stretched
    }
    double clamp_parameter(double p) const override {
        return snap_r(std::clamp(p, kH2pRMin, kH2pRMax));
    }
    void geometry_changed() override {
        atlas_.clear();  // a new R needs a fresh atlas
        atlas_R_ = -1.0;
        rebuild_markers();
    }

    std::string title_suffix() override {
        const double rep = nuclear_repulsion();
        std::string s = strf("  R = %.3f Bohr (fixed nuclei)", snap_r(param_));
        if (!atlas_.empty()) {
            s += strf("  E_total(1sigma_g) = %.4f Ha",
                      atlas_[0].orb.energy + rep);
        }
        if (showing_random_) {
            s += "  showing: random atlas superposition";
        } else if (cur_ >= 0 && cur_ < static_cast<int>(atlas_.size())) {
            s += strf("  showing %s: E_elec = %.4f Ha",
                      atlas_[static_cast<std::size_t>(cur_)].label.c_str(),
                      atlas_[static_cast<std::size_t>(cur_)].orb.energy);
        }
        s += strf("  (%d orbitals)  keys: 2.. orbitals / S random / R slider",
                  static_cast<int>(atlas_.size()));
        return s;
    }

private:
    static ses::WavepacketSimulation make(double r) {
        const ses::Grid1D axis{-kH2pBox, kH2pBox, kH2pPoints};
        const ses::Grid3D grid{axis, axis, axis};
        const double d = 0.5 * snap_r(r);
        return ses::WavepacketSimulation{ses::WavepacketSimulation::Config{
            grid,
            ses::regularized_coulomb_potential(
                grid, 1.0, {{-d, 0.0, 0.0}, {d, 0.0, 0.0}}),
            ses::Vec3d{},
            ses::Vec3d{1.8, 1.8, 1.8},
            ses::Vec3d{},
            kH2pDt,
        }};
    }

    // Snap the HALF-distance to a grid point so both regularized nucleus
    // cells stay honest (R in multiples of 2h).
    static double snap_r(double r) {
        const double h = 2.0 * kH2pBox / kH2pPoints;
        const double m =
            std::max(1.0, std::round(std::clamp(r, kH2pRMin, kH2pRMax) /
                                     (2.0 * h)));
        return 2.0 * h * m;
    }

    void rebuild_markers() {
        // Two protons as CPK hydrogen-white balls.
        const std::vector<ses::Vec3d> c = centers();
        balls_ = {ball(c[0], 0.4, 0.95f, 0.95f, 0.95f),
                  ball(c[1], 0.4, 0.95f, 0.95f, 0.95f)};
    }

    // One exposed orbital = an analytic MO + which real partner (m>0 has a
    // cos-phi and a sin-phi lobing) + its term-symbol label.
    struct Mo {
        ses::H2plusOrbital orb;
        int partner;
        std::string label;
    };

    // Drop a synthesized/blended field in as the live state and evolve it.
    void set_state_field(const ses::Field3D& psi) {
        sim_.set_psi(psi);
        cpu_is_truth_ = true;  // run_frame uploads it to the engine
        stepping_ = BaseStepping::RealTime;
        title_dirty_ = true;
        stage_active_view();
    }

    // Radial extent test: an orbital is representable if its Lambda(xi) has
    // decayed well inside the box (physical r ~ (R/2) xi).
    bool representable(const ses::H2plusOrbital& o, double R) const {
        double peak = 0.0;
        for (double v : o.lambda) {
            peak = std::max(peak, v * v);
        }
        if (peak <= 0.0) {
            return false;
        }
        double xi_dec = o.xi.empty() ? 1.0 : o.xi.back();
        for (std::size_t i = o.lambda.size(); i-- > 0;) {
            if (o.lambda[i] * o.lambda[i] > 1e-4 * peak) {
                xi_dec = o.xi[i];
                break;
            }
        }
        return 0.5 * R * xi_dec < 0.85 * kH2pBox;
    }

    static std::string mo_label(const ses::H2plusOrbital& o, int partner) {
        // United-atom / MO term symbol from (m, n_eta, n_xi, parity).
        const char* g = o.parity > 0 ? "g" : "u";
        const char* sym = o.m == 0 ? "sigma" : (o.m == 1 ? "pi" : "delta");
        // Principal-ish index within the (m,parity) tower for a readable name.
        std::string s = strf("%d%s_%s%s", o.n_xi + o.n_eta + o.m + 1, sym, g,
                             o.parity < 0 && o.m == 0 ? "*" : "");
        if (o.m > 0) {
            s += partner == 0 ? " (y)" : " (z)";
        }
        return s;
    }

    void ensure_atlas() const {
        const double R = snap_r(param_);
        if (atlas_R_ == R && !atlas_.empty()) {
            return;
        }
        atlas_.clear();
        atlas_R_ = R;
        // Load the BAKED atlas (ses.h2plus_atlas_data, offline-generated by
        // sesolver_genatlas): the exact prolate-spheroidal orbitals for the
        // nearest snapped R -- zero runtime ODE solve.
        const std::vector<ses::H2plusOrbital> orbs = ses::h2plus_atlas_baked(R);
        for (const ses::H2plusOrbital& o : orbs) {
            if (!representable(o, R)) {
                continue;
            }
            const int partners = o.m > 0 ? 2 : 1;
            for (int pp = 0; pp < partners; ++pp) {
                atlas_.push_back({o, pp, mo_label(o, pp)});
            }
        }
    }

    // Mutable analytic-atlas cache (rebuilt on R change; a const lazy build).
    mutable std::vector<Mo> atlas_;
    mutable double atlas_R_ = -1.0;
    int cur_ = 0;  // currently-shown atlas index
};

// ---- Benzene one-electron toy --------------------------------------------

constexpr double kBzBox = 12.0;   // Bohr half-extent, 256^3 (h ~ 0.094)
constexpr int kBzPoints = 256;
// dt scaled to the Z=6 regularized well: the nucleus cell sits at
// V = -Z*2.38/h ~ -152 Ha, and the half-potential phase V*dt/2 must stay
// well under a radian per step or the Trotter product heats the state over
// the whole box (P(r<6) collapsed to 0.09 within 5 au at dt = 0.04).
// 0.004 -> 0.30 rad at the deepest cell; the core-band eigenphase
// |E|*dt ~ 0.23 rad rides along.
constexpr double kBzDt = 0.004;
constexpr double kBzRingR = 2.63;   // C-C 1.39 A in bohr
constexpr double kBzCH = 2.06;      // C-H 1.09 A in bohr: H at r + this
// BARE nuclear charges of the stripped molecule -- nothing to calibrate.
constexpr double kBzZC = 6.0;
constexpr double kBzZH = 1.0;

class BenzeneDirector final : public MoleculeDirectorBase {
public:
    BenzeneDirector() : MoleculeDirectorBase(make()) {
        rebuild_markers();
    }

    // The REAL benzene geometry only (uniform ring, as X-ray settled) --
    // no counterfactual knobs in this simulator.
    void set_geometry(int /*variant*/) override {}
    void set_parameter(double /*p*/) override {}

    // The deep carbon-core band: ground + two deflated members (keys 2/3/4).
    int exposed_states() const override { return 3; }

    double default_camera_azimuth() const override { return 0.3; }
    double default_camera_elevation() const override { return 0.7; }
    double default_camera_distance() const override { return 40.0; }

    // Ball-and-stick: the neutral gray bond skeleton as an overlay line,
    // the atoms themselves as CPK marker balls (carbon black, hydrogen
    // white).
    int overlay_curve_count() const override { return 1; }
    OverlayCurve overlay_curve(int /*i*/) const override {
        // bonds (hexagon + C-H spokes)
        return {ring_marker_.data(),
                static_cast<int>(ring_marker_.size() / 3),
                0.55f, 0.55f, 0.60f, 0.9f};
    }

protected:
    const char* scene_name() const override {
        return "Stripped benzene (first electron over bare nuclei)";
    }

    ses::WavepacketSimulation remake_simulation() const override {
        return make();
    }

    // Carbons first, then their hydrogens (each riding its carbon's angle);
    // every center lattice-snapped for the bare-cell regularization.
    std::vector<ses::Vec3d> centers() const override {
        std::vector<ses::Vec3d> c = snapped_ring(sim_.grid(), kBzRingR);
        const std::vector<ses::Vec3d> h =
            snapped_ring(sim_.grid(), kBzRingR + kBzCH);
        c.insert(c.end(), h.begin(), h.end());
        return c;
    }
    std::vector<double> charges() const override {
        std::vector<double> q(12, kBzZC);
        for (int i = 6; i < 12; ++i) {
            q[static_cast<std::size_t>(i)] = kBzZH;
        }
        return q;
    }
    // In-plane displaced seeds pick up the ring-momentum pair.
    ses::Field3D excited_seed(int k) const override {
        const ses::Vec3d at = k == 1
                                  ? ses::Vec3d{kBzRingR, 0.4, 0.0}
                                  : ses::Vec3d{-0.5, kBzRingR, 0.0};
        return ses::gaussian_wavepacket(sim_.grid(), at,
                                        ses::Vec3d{1.5, 1.5, 1.2},
                                        ses::Vec3d{});
    }
    double geometry_parameter(int /*variant*/) const override { return 0.0; }
    double clamp_parameter(double /*p*/) const override { return 0.0; }
    void geometry_changed() override { rebuild_markers(); }

    std::string title_suffix() override {
        std::string s{"  uniform ring (the X-ray geometry)"};
        if (prepared_[0]) {
            s += strf("  E0 = %.4f", e_[0]);
        }
        if (prepared_[1]) {
            s += strf("  E1 = %.4f", e_[1]);
        }
        if (prepared_[2]) {
            s += strf("  E2 = %.4f (quasi-degenerate carbon-core band)",
                      e_[2]);
        }
        s += "  BARE nuclei: 6 C (Z=6) + 6 H (Z=1), regularized cells, "
             "lattice-snapped; C6H6^41+ first electron.  keys: 2/3/4 states";
        return s;
    }

private:
    static std::vector<ses::Vec3d> snapped_ring(const ses::Grid3D& g,
                                                double radius) {
        const double kPi = 3.14159265358979323846;
        std::vector<ses::Vec3d> c;
        for (int i = 0; i < 6; ++i) {
            const double th = kPi / 3.0 * i;
            c.push_back(ses::snap_to_grid(
                g, {radius * std::cos(th), radius * std::sin(th), 0.0}));
        }
        return c;
    }

    static ses::WavepacketSimulation make() {
        const ses::Grid1D axis{-kBzBox, kBzBox, kBzPoints};
        const ses::Grid3D grid{axis, axis, axis};
        std::vector<double> v = ses::regularized_coulomb_potential(
            grid, kBzZC, snapped_ring(grid, kBzRingR));
        const std::vector<double> vh = ses::regularized_coulomb_potential(
            grid, kBzZH, snapped_ring(grid, kBzRingR + kBzCH));
        for (std::size_t i = 0; i < v.size(); ++i) {
            v[i] += vh[i];
        }
        return ses::WavepacketSimulation{ses::WavepacketSimulation::Config{
            grid,
            std::move(v),
            ses::Vec3d{},
            ses::Vec3d{1.8, 1.8, 1.2},
            ses::Vec3d{},
            kBzDt,
        }};
    }

    // One retraced strip: the C hexagon with a C->H spoke drawn (and
    // retraced) at every vertex -- the full skeleton in one overlay curve.
    void rebuild_markers() {
        const ses::Grid1D axis{-kBzBox, kBzBox, kBzPoints};
        const ses::Grid3D grid{axis, axis, axis};
        const std::vector<ses::Vec3d> c = snapped_ring(grid, kBzRingR);
        const std::vector<ses::Vec3d> h =
            snapped_ring(grid, kBzRingR + kBzCH);
        ring_marker_.clear();
        auto put = [&](const ses::Vec3d& p) {
            ring_marker_.push_back(static_cast<float>(p.x));
            ring_marker_.push_back(static_cast<float>(p.y));
            ring_marker_.push_back(static_cast<float>(p.z));
        };
        for (int i = 0; i < 6; ++i) {
            put(c[static_cast<std::size_t>(i)]);
            put(h[static_cast<std::size_t>(i)]);
            put(c[static_cast<std::size_t>(i)]);
        }
        put(c[0]);  // close the hexagon
        // CPK atom balls: carbon larger and dark, hydrogen smaller, white.
        balls_.clear();
        for (int i = 0; i < 6; ++i) {
            balls_.push_back(
                ball(c[static_cast<std::size_t>(i)], 0.55, 0.22f, 0.22f,
                     0.25f));
        }
        for (int i = 0; i < 6; ++i) {
            balls_.push_back(
                ball(h[static_cast<std::size_t>(i)], 0.38, 0.95f, 0.95f,
                     0.95f));
        }
    }

    std::vector<float> ring_marker_;
};

}  // namespace ses_shell
