module;
#include <complex>
#include <array>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>
export module ses.app.atom_model;
export import ses.app.manifold_spec;
export import ses.vk.engine;
export import ses.radial;
export import ses.decay;
export import ses.vram_budget;


// The tracked-atom model: the radial solve, the on-demand eigenstate
// synthesis bookkeeping (energies, grid norms, resident handles, fp16
// policy), and the E1 decay channel table. Engine-backed operations take
// ses_vk::Engine by reference; everything else is pure data/logic. UI
// concerns (titles, timers, queues' frame pacing) stay in the shell.


export namespace ses_shell {

class AtomModel {
public:
    // Solve the radial atom once (blocking, well under a second): in-box
    // levels (u(R_box) = 0) back the tracked manifold; the free-atom table
    // to n = 10 is printed for the record.
    void solve_radial_atom(double r_box) {
        radial_grid_ = ses::RadialGrid{r_box, 5119};
        std::vector<double> v(static_cast<std::size_t>(radial_grid_.n));
        for (int i = 0; i < radial_grid_.n; ++i) {
            v[static_cast<std::size_t>(i)] = -1.0 / radial_grid_.r(i);  // r=(i+1)h>0
        }
        for (int lev = 0; lev < kNumLevels; ++lev) {
            const ses::RadialState st = ses::radial_eigenstate(
                radial_grid_,
                ses::radial_hamiltonian(radial_grid_, v, kLevelSpec[lev].l),
                kLevelSpec[lev].k);
            radial_u_[static_cast<std::size_t>(lev)] = st.u;
            radial_energy_[static_cast<std::size_t>(lev)] = st.energy;
        }
        radial_ready_ = true;

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
        const StateSpec& sp = kStateSpec[s];
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
        const StateSpec& sp = kStateSpec[static_cast<std::size_t>(idx)];
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
        const StateSpec& sp = kStateSpec[static_cast<std::size_t>(idx)];
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
        const StateSpec& sp = kStateSpec[static_cast<std::size_t>(idx)];
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
        const StateSpec& sp = kStateSpec[s];
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
        const StateSpec& sp = kStateSpec[s];
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
        const StateSpec& sp = kStateSpec[s];
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

    // Downward pairs worth a dipole integral: gap > 1e-3 skips degenerate
    // m-splittings and sub-mHa channels; |dl| = 1 and the real-basis m rule
    // apply the E1 selection rules analytically. In the tesseral basis the
    // phi integral makes z couple m' == m only and x/y couple
    // ||m'| - |m|| == 1 only -- everything else (including m' == -m) is
    // EXACTLY zero, so those integrals never reach the GPU.
    void collect_channel_pairs() {
        pair_queue_.clear();
        for (int from = 0; from < kNumStates; ++from) {
            for (int to = 0; to < kNumStates; ++to) {
                const bool downward =
                    state_energy_[static_cast<std::size_t>(from)] -
                        state_energy_[static_cast<std::size_t>(to)] >
                    1e-3;
                const bool dl_allowed =
                    std::abs(kStateSpec[from].l - kStateSpec[to].l) == 1;
                const int mf = kStateSpec[from].m;
                const int mt = kStateSpec[to].m;
                const bool dm_allowed =
                    mt == mf || std::abs(std::abs(mt) - std::abs(mf)) == 1;
                if (downward && dl_allowed && dm_allowed) {
                    pair_queue_.push_back({from, to});
                }
            }
        }
    }

    std::vector<std::pair<int, int>>& pair_queue() { return pair_queue_; }

    void evaluate_channel_pair(ses_vk::Engine& engine,
                               const std::pair<int, int>& p) {
        const std::size_t from = static_cast<std::size_t>(p.first);
        const std::size_t to = static_cast<std::size_t>(p.second);
        const double gap = state_energy_[from] - state_energy_[to];
        // Transient endpoints: cache the 'from' orbital across its
        // consecutive channels, synthesize each 'to' fresh -- peak residency
        // is 2 orbitals.
        if (pair_from_idx_ != p.first) {
            if (pair_from_buf_ >= 0) {
                engine.release_state(pair_from_buf_);
            }
            pair_from_buf_ = synth_transient(engine, p.first);
            pair_from_idx_ = p.first;
        }
        const int to_buf = synth_transient(engine, p.second);
        const ses::DipoleMatrixElement d = engine.dipole_between(to_buf, pair_from_buf_);
        engine.release_state(to_buf);
        channels_.push_back(ShellChannel{
            p.first, p.second, ses::einstein_a(gap, ses::dipole_strength_sq(d)), 0.0});
        if (p.first == kP2Z && p.second == kS1) {
            dipole_z_ = std::abs(d.z);  // laser E0 drive strength, ready for free
        }
    }

    // Free the channel-build 'from' cache (the finale of the chunked build).
    void release_pair_cache(ses_vk::Engine& engine) {
        if (pair_from_buf_ >= 0) {
            engine.release_state(pair_from_buf_);
            pair_from_buf_ = -1;
        }
        pair_from_idx_ = -1;
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
        for (int s = 0; s < kNumStates; ++s) {
            std::fprintf(stderr, "manifold: E(%s) = %.6f Ha\n", kStateSpec[s].name,
                         state_energy_[static_cast<std::size_t>(s)]);
        }
        for (const ShellChannel& c : channels_) {
            std::fprintf(stderr, "manifold: %s -> %s  A = %.3e /au  tau = %.3e au%s\n",
                         kStateSpec[c.from].name, kStateSpec[c.to].name, c.a_true,
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
        // The whole blocking build: dipoles reduce on the GPU and never
        // touch the CPU; the engine owns its frames.
        bool ok = true;
        for (int idx = 0; idx < kNumStates && ok; ++idx) {
            ok = ensure_state(engine, idx);
        }
        if (ok) {
            collect_channel_pairs();
            while (!pair_queue_.empty()) {
                evaluate_channel_pair(engine, pair_queue_.back());
                pair_queue_.pop_back();
            }
        }
        if (!ok) {
            return false;
        }
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

    // E1 decay channel table + the chunked-build work queue.
    std::vector<ShellChannel> channels_;
    std::vector<std::pair<int, int>> pair_queue_;
    double accel_display_ = 0.0;  // common display acceleration factor
    int pair_from_idx_ = -1;      // channel-build 'from' cache (transient)
    int pair_from_buf_ = -1;
    double dipole_z_ = 0.0;  // |<2p_z| z |1s>|, the laser drive strength
};

}  // namespace ses_shell
