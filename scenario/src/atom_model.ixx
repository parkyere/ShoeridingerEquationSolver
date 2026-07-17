module;
#include <complex>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
export module ses.scenario.atom_model;
export import ses.scenario.manifold_spec;
export import ses.vk.engine;
export import ses.radial;
export import ses.decay;
export import ses.vram_budget;
import ses.harmonics;


// The tracked-manifold model for a CENTRAL potential: the radial solve, the
// on-demand eigenstate synthesis bookkeeping (energies, grid norms, resident
// handles, fp16 policy), and the E1 decay channel table. Hydrogen passes the
// bare -1/r + kStateSpec; any central scene (harmonic trap) passes its own
// spec + radial potential -- the machinery is identical. Engine-backed
// operations take ses_vk::Engine by reference; everything else is pure
// data/logic. UI concerns (titles, timers, frame pacing) stay in the shell.


export namespace ses_shell {

class AtomModel {
public:
    // Solve the radial levels of a central potential (values vr on grid rg)
    // and adopt the given manifold spec. SPEC CONVENTION: index 0 is the
    // ground state; if a p_z-like state exists at index kP2Z the laser drive
    // strength is captured for free (hydrogen's frozen first-five layout).
    // Capacities are the hydrogen spec's (kNumStates/kNumLevels).
    void solve_radial_manifold(const ses::RadialGrid& rg,
                               const std::vector<double>& vr,
                               const StateSpec* states, int n_states,
                               const RadialLevelSpec* levels, int n_levels) {
        assert(n_states <= kNumStates && n_levels <= kNumLevels);
        spec_ = states;
        n_states_ = n_states;
        radial_grid_ = rg;
        for (int lev = 0; lev < n_levels; ++lev) {
            const ses::RadialState st = ses::radial_eigenstate(
                radial_grid_,
                ses::radial_hamiltonian(radial_grid_, vr, levels[lev].l),
                levels[lev].k);
            radial_u_[static_cast<std::size_t>(lev)] = st.u;
            radial_energy_[static_cast<std::size_t>(lev)] = st.energy;
        }
        radial_ready_ = true;
    }

    int n_states() const { return n_states_; }
    const StateSpec& state_spec(int idx) const {
        return spec_[static_cast<std::size_t>(idx)];
    }

    // Hydrogen: bare -1/r levels + the kStateSpec manifold (blocking, well
    // under a second), plus the free-atom lifetime table for the record.
    void solve_radial_atom(double r_box) {
        const ses::RadialGrid rg{r_box, 5119};
        std::vector<double> v(static_cast<std::size_t>(rg.n));
        for (int i = 0; i < rg.n; ++i) {
            v[static_cast<std::size_t>(i)] = -1.0 / rg.r(i);  // r=(i+1)h>0
        }
        solve_radial_manifold(rg, v, kStateSpec, kNumStates, kLevelSpec,
                              kNumLevels);

        // The full free-atom lifetime table to n = 10 (55 levels).
        const ses::RadialGrid free_grid{600.0, 14999};
        std::vector<double> vf(static_cast<std::size_t>(free_grid.n));
        for (int i = 0; i < free_grid.n; ++i) {
            vf[static_cast<std::size_t>(i)] = -1.0 / free_grid.r(i);
        }
        const std::vector<ses::LevelInfo> table =
            ses::bound_level_table(free_grid, vf, 10);
        std::fprintf(stderr,
                     "spectrum: free hydrogen atom (true -1/r), ALL %d bound levels to "
                     "n = 10 (E1 lifetimes from our radial engine)\n",
                     static_cast<int>(table.size()));
        const char* kSpdf = "spdfghijkl";
        for (const ses::LevelInfo& e : table) {
            std::fprintf(stderr,
                         "spectrum: %2d%c  E = %11.6f Ha   tau = %.3e au (%.3e ns)%s\n",
                         e.n, kSpdf[e.l], e.energy, e.lifetime,
                         e.lifetime * 2.4188843e-17 * 1e9,
                         e.lifetime == 0.0 ? "  [E1-stable]" : "");
        }
    }

    bool radial_ready() const { return radial_ready_; }
    const ses::RadialGrid& radial_grid() const { return radial_grid_; }

    // Atlas storage precision, decided at startup from free VRAM.
    void set_precision(ses::GpuPrecision p) { precision_ = p; }
    ses::GpuPrecision precision() const { return precision_; }

    // Whether tracked state `idx` is stored fp16. Fp32 is kept for the h-audit
    // s-states (E_radial vs <H>_grid amplifies fp16 noise through the
    // Laplacian) and the deflation set (read as fp32 phi buffers).
    bool state_is_fp16(int idx) const {
        if (precision_ != ses::GpuPrecision::Fp16) {
            return false;
        }
        switch (idx) {
            case kS1:
            case kP2X:
            case kP2Y:
            case kP2Z:
            case k4S:
            case k5S:
            case k6S:
                return false;
            default:
                return true;
        }
    }

    // The engine handle for tracked state `idx` (precision lives inside the
    // engine's state record, so an int handle carries both).
    int handle(int idx) const { return state_buf_[static_cast<std::size_t>(idx)]; }

    double state_energy(int idx) const {
        return state_energy_[static_cast<std::size_t>(idx)];
    }

    int gpu_synthesize(ses_vk::Engine& engine, int idx,
                       double* out_peak = nullptr) {
        const std::size_t s = static_cast<std::size_t>(idx);
        const StateSpec& sp = spec_[s];
        state_energy_[s] = radial_energy_[static_cast<std::size_t>(sp.level)];
        if (state_is_fp16(idx)) {
            return engine.synthesize_state_half(
                radial_u_[static_cast<std::size_t>(sp.level)], sp.l, sp.m,
                radial_grid_.h(), radial_grid_.rmax, radial_grid_.n, out_peak,
                &state_norm2_[s]);
        }
        return engine.synthesize_state(
            radial_u_[static_cast<std::size_t>(sp.level)], sp.l, sp.m,
            radial_grid_.h(), radial_grid_.rmax, radial_grid_.n, out_peak,
            &state_norm2_[s]);
    }

    // Normalized complex amplitude <n|psi> from the last project_psi() pass
    // (the MCWF no-jump damping consumes it; population = |amplitude|^2).
    std::complex<double> project_state_amplitude(const ses_vk::Engine& engine,
                                                 int idx) const {
        const StateSpec& sp = spec_[static_cast<std::size_t>(idx)];
        const double n2 = state_norm2_[static_cast<std::size_t>(idx)];
        if (n2 <= 0.0) {
            return {};
        }
        return engine.project_amplitude(
                   radial_u_[static_cast<std::size_t>(sp.level)], sp.l, sp.m) /
               std::sqrt(n2);
    }

    // Population |<n|psi>|^2 from the last engine.project_psi() pass: the 1-D
    // radial dot g_lm . u_nl, grid-normalized so the value equals a full 3D
    // inner product against the grid-normalized orbital.
    double project_population(const ses_vk::Engine& engine, int idx) const {
        const StateSpec& sp = spec_[static_cast<std::size_t>(idx)];
        const double n2 = state_norm2_[static_cast<std::size_t>(idx)];
        if (n2 <= 0.0) {
            return 0.0;
        }
        return std::norm(engine.project_amplitude(
                   radial_u_[static_cast<std::size_t>(sp.level)], sp.l, sp.m)) /
               n2;
    }

    // Collapse psi onto eigenstate `idx` by synthesizing it ON DEMAND (no
    // resident atlas): psi is overwritten with the normalized orbital.
    void collapse_onto(ses_vk::Engine& engine, int idx) {
        const StateSpec& sp = spec_[static_cast<std::size_t>(idx)];
        engine.synthesize_into_psi(radial_u_[static_cast<std::size_t>(sp.level)],
                                   sp.l, sp.m, radial_grid_.h(), radial_grid_.rmax,
                                   radial_grid_.n);
    }

    // Synthesize eigenstate idx into a FRESH fp32 buffer (caller frees);
    // captures its grid norm (for populations) and energy. All
    // build/deflation work uses these transients -- no resident atlas.
    int synth_transient(ses_vk::Engine& engine, int idx,
                        double* out_peak = nullptr) {
        const std::size_t s = static_cast<std::size_t>(idx);
        const StateSpec& sp = spec_[s];
        state_energy_[s] = radial_energy_[static_cast<std::size_t>(sp.level)];
        return engine.synthesize_state(radial_u_[static_cast<std::size_t>(sp.level)],
                                       sp.l, sp.m, radial_grid_.h(), radial_grid_.rmax,
                                       radial_grid_.n, out_peak, &state_norm2_[s]);
    }

    // Fused-MCWF term for tracked state idx with coefficient d: radial table
    // + (l, m) + the cached grid-norm normalizer (a constant per state).
    // False when the cache is cold -- the caller falls back to synth_over.
    bool mcwf_term(int idx, std::complex<double> d,
                   ses_vk::Engine::McwfTerm* out) const {
        const std::size_t s = static_cast<std::size_t>(idx);
        const StateSpec& sp = spec_[s];
        const double n2 = state_norm2_[s];
        if (n2 <= 0.0) {
            return false;
        }
        *out = {&radial_u_[static_cast<std::size_t>(sp.level)], sp.l, sp.m,
                d.real(), d.imag(), 1.0 / std::sqrt(n2)};
        return true;
    }

    // Re-synthesize eigenstate idx into an existing scratch buffer (the MCWF
    // path refills one buffer 8x per title tick; fresh transients would
    // device-idle + reallocate each time). Bookkeeping = synth_transient.
    bool synth_over(ses_vk::Engine& engine, int handle, int idx) {
        const std::size_t s = static_cast<std::size_t>(idx);
        const StateSpec& sp = spec_[s];
        state_energy_[s] = radial_energy_[static_cast<std::size_t>(sp.level)];
        return engine.synthesize_state_over(
            handle, radial_u_[static_cast<std::size_t>(sp.level)], sp.l, sp.m,
            radial_grid_.h(), radial_grid_.rmax, radial_grid_.n,
            &state_norm2_[s]);
    }

    // Ensure state `idx` has a resident GPU buffer, synthesized on the GPU.
    bool ensure_state(ses_vk::Engine& engine, int idx) {
        const std::size_t s = static_cast<std::size_t>(idx);
        if (state_buf_[s] >= 0) {
            return true;
        }
        if (!radial_ready_) {
            return false;
        }
        state_buf_[s] = gpu_synthesize(engine, idx);
        return state_buf_[s] >= 0;
    }

    // The E1 channel table, factorized: for (u/r) Y_lm states the 3D dipole
    // integral separates EXACTLY into (1D radial integral) x (constexpr
    // tesseral angular strength, ses.harmonics), so the whole table is ~40
    // radial dot products on the CPU -- no GPU pass. Gap > 1e-3 skips the
    // degenerate intra-shell channels; the |dl| = 1 and tesseral m rules
    // (m' == -m exactly forbidden) are hard zeros of tesseral_e1_sq.
    void build_channel_table() {
        channels_.clear();
        std::array<std::array<double, kNumLevels>, kNumLevels> rint{};
        std::array<std::array<bool, kNumLevels>, kNumLevels> have{};
        for (int s = 0; s < n_states_; ++s) {
            state_energy_[static_cast<std::size_t>(s)] = radial_energy_[
                static_cast<std::size_t>(spec_[s].level)];
        }
        for (int from = 0; from < n_states_; ++from) {
            const StateSpec& sf = spec_[from];
            for (int to = 0; to < n_states_; ++to) {
                const StateSpec& st = spec_[to];
                const double gap =
                    state_energy_[static_cast<std::size_t>(from)] -
                    state_energy_[static_cast<std::size_t>(to)];
                if (gap <= 1e-3) {
                    continue;
                }
                const double angular =
                    ses::tesseral_e1_sq(st.l, st.m, sf.l, sf.m);
                if (angular == 0.0) {
                    continue;
                }
                const std::size_t lf = static_cast<std::size_t>(sf.level);
                const std::size_t lt = static_cast<std::size_t>(st.level);
                if (!have[lf][lt]) {
                    rint[lf][lt] = ses::radial_dipole_integral(
                        radial_grid_, radial_u_[lt], radial_u_[lf]);
                    have[lf][lt] = true;
                }
                const double r = rint[lf][lt];
                channels_.push_back(ShellChannel{
                    from, to, ses::einstein_a(gap, r * r * angular), 0.0});
                if (from == kP2Z && to == kS1) {
                    // Laser E0 drive strength |<1s|z|2p_z>| = |R| sqrt(1/3).
                    dipole_z_ = std::abs(r) *
                                std::sqrt(ses::tesseral_e1_axis_sq(2, 0, 0, 1, 0));
                }
            }
        }
    }

    bool finalize_channel_table(double gamma_display_target) {
        double a_max = 0.0;
        for (const ShellChannel& c : channels_) {
            a_max = std::max(a_max, c.a_true);
        }
        if (a_max <= 0.0) {
            channels_.clear();
            return false;
        }
        accel_display_ = gamma_display_target / a_max;
        for (ShellChannel& c : channels_) {
            c.gamma_display = c.a_true * accel_display_;
        }
        std::fprintf(stderr, "manifold: display acceleration x%.3e\n", accel_display_);
        for (int s = 0; s < n_states_; ++s) {
            std::fprintf(stderr, "manifold: E(%s) = %.6f Ha\n", spec_[s].name,
                         state_energy_[static_cast<std::size_t>(s)]);
        }
        for (const ShellChannel& c : channels_) {
            std::fprintf(stderr, "manifold: %s -> %s  A = %.3e /au  tau = %.3e au%s\n",
                         spec_[c.from].name, spec_[c.to].name, c.a_true,
                         c.a_true > 0.0 ? 1.0 / c.a_true : 0.0,
                         c.a_true < 1e-3 * a_max ? "  [forbidden/suppressed]" : "");
        }
        return true;
    }

    const std::vector<ShellChannel>& channels() const { return channels_; }
    double accel_display() const { return accel_display_; }
    double dipole_z() const { return dipole_z_; }

    // Total decay lifetime of a tracked state (sum over its channels), in
    // TRUE atomic units; 0 means no open channel (stable/metastable).
    double lifetime_of(int state) const {
        double a_sum = 0.0;
        for (const ShellChannel& c : channels_) {
            if (c.from == state) {
                a_sum += c.a_true;
            }
        }
        return a_sum > 0.0 ? 1.0 / a_sum : 0.0;
    }

    // Blocking fallbacks over synthesis: no-ops once the startup build has
    // the channel table. The engine drives its own offscreen frames, so they
    // are legal any time between paints.
    bool prepare_ground_cache(ses_vk::Engine& engine) {
        return ensure_state(engine, kS1);
    }

    // The laser pair (1s + 2p_z); dipole_z_ (the drive strength) comes
    // from the channel table or is computed here if the table is not up.
    bool prepare_excited_cache(ses_vk::Engine& engine) {
        const bool ok = ensure_state(engine, kS1) && ensure_state(engine, kP2Z);
        if (ok && dipole_z_ == 0.0) {
            const ses::DipoleMatrixElement d =
                engine.dipole_between(handle(kP2Z), handle(kS1));
            dipole_z_ = std::abs(d.z);
        }
        return ok;
    }

    bool prepare_p_triplet(ses_vk::Engine& engine) {
        if (!prepare_excited_cache(engine)) {
            return false;
        }
        return ensure_state(engine, kP2X) && ensure_state(engine, kP2Y);
    }

    bool prepare_manifold_cache(ses_vk::Engine& engine,
                                double gamma_display_target) {
        if (!channels_.empty()) {
            return true;
        }
        // Synthesis captures each state's grid norm (state_norm2_), which
        // projections and the MCWF terms divide by; the channel table itself
        // is the factorized CPU build.
        bool ok = true;
        for (int idx = 0; idx < n_states_ && ok; ++idx) {
            ok = ensure_state(engine, idx);
        }
        if (!ok) {
            return false;
        }
        build_channel_table();
        return finalize_channel_table(gamma_display_target);
    }

private:
    static std::array<int, kNumStates> make_unset_handles() {
        std::array<int, kNumStates> a{};
        a.fill(-1);
        return a;
    }

    // Radial solve products (the 1-D atom).
    ses::RadialGrid radial_grid_{};
    // The adopted manifold spec (defaults: hydrogen); arrays below are
    // hydrogen-capacity, the first n_states_/levels entries are live.
    const StateSpec* spec_ = kStateSpec;
    int n_states_ = kNumStates;

    std::array<std::vector<double>, kNumLevels> radial_u_;
    std::array<double, kNumLevels> radial_energy_{};
    bool radial_ready_ = false;

    // Tracked-manifold bookkeeping: resident handles (-1 = not resident),
    // per-state energies, pre-normalization grid norms (the projection
    // population normalizer), and the storage-precision policy.
    std::array<int, kNumStates> state_buf_ = make_unset_handles();
    std::array<double, kNumStates> state_norm2_{};
    std::array<double, kNumStates> state_energy_{};
    ses::GpuPrecision precision_ = ses::GpuPrecision::Fp32;

    // E1 decay channel table (factorized CPU build).
    std::vector<ShellChannel> channels_;
    double accel_display_ = 0.0;  // common display acceleration factor
    double dipole_z_ = 0.0;  // |<2p_z| z |1s>|, the laser drive strength
};

}  // namespace ses_shell
