module;
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
export module ses.scenario.harmonic_director;
export import ses.scenario.base_director;
import ses.scenario.atom_model;
import ses.projection;
import ses.measurement;


// The 3D isotropic harmonic trap scenario. Two complementary demos in one
// scene (the project's first physics ruling, live):
//  - CLASSICAL: a coherent state (the ground Gaussian displaced by x0)
//    oscillates rigidly at omega and radiates the continuous Larmor power
//    the title shows (Key 2 relaxes to the ground state; R re-displaces).
//  - QED: the trap is a CENTRAL potential, so the atom's tracked-manifold
//    machinery applies verbatim (license: tests/trap_ladder_test.cpp) --
//    Key 5 prepares a Fock-ladder eigenstate (static, Larmor ~0), Key D
//    arms Einstein-A quantum jumps that cascade N -> N-1, every photon at
//    exactly omega (r is a ladder operator).


export namespace ses_shell {

constexpr double kTrapOmega = 0.25;      // au; period 2 pi / w ~ 25 au
constexpr double kTrapBox = 20.0;        // Bohr half-extent (h = 0.15625)
constexpr double kCoherentOffset = 8.0;  // Bohr; classical turning point

// Tracked ladder manifold: N = 2k + l <= 3 -- 20 tesseral states over 6
// radial levels, named by N (E = (N + 3/2) w). Follows the spec convention
// (ground at 0, p_z at kP2Z, second s at 4) so AtomModel's ground/laser
// index conventions carry.
inline constexpr int kNumTrapLevels = 6;
inline constexpr RadialLevelSpec kTrapLevels[kNumTrapLevels] = {
    {0, 0}, {1, 0}, {0, 1}, {2, 0}, {1, 1}, {3, 0},
};
inline constexpr int kNumTrapStates = 20;
inline constexpr StateSpec kTrapStates[kNumTrapStates] = {
    {0, 0, 0, "0s"},
    {1, 1, 1, "1p_x"}, {1, 1, -1, "1p_y"}, {1, 1, 0, "1p_z"},
    {2, 0, 0, "2s"},
    {3, 2, -2, "2d_xy"}, {3, 2, -1, "2d_yz"}, {3, 2, 0, "2d_z2"},
    {3, 2, 1, "2d_zx"}, {3, 2, 2, "2d_x2y2"},
    {4, 1, 1, "3p_x"}, {4, 1, -1, "3p_y"}, {4, 1, 0, "3p_z"},
    {5, 3, -3, "3f_-3"}, {5, 3, -2, "3f_-2"}, {5, 3, -1, "3f_-1"},
    {5, 3, 0, "3f_0"}, {5, 3, 1, "3f_+1"}, {5, 3, 2, "3f_+2"},
    {5, 3, 3, "3f_+3"},
};

// Display decay rate target (tau_display ~ 8 au, as the atom) and the
// post-collapse flush budgets (contracts: eigenstate_flush_test).
constexpr double kTrapGammaDisplay = 0.125;
constexpr int kTrapFlushSteps = 6;
constexpr int kTrapFlushStepsGround = 24;  // 0s is the ITP fixed point
constexpr int kTrapFlashTicks = 25;

class HarmonicDirector final : public BaseDirector {
public:
    HarmonicDirector() : BaseDirector(make()) {}

    bool handle_key(char key) override {
        if (BaseDirector::handle_key(key)) {
            return true;
        }
        switch (key) {
            case '5':
                excite_next();
                return true;
            case 'D':
                toggle_decay();
                return true;
            case 'E':
                measure_energy_now();
                return true;
            default:
                return false;
        }
    }

    float next_flash_intensity() override {
        if (flash_ticks_ <= 0) {
            return 0.0f;
        }
        const float w = static_cast<float>(flash_ticks_) /
                        static_cast<float>(kTrapFlashTicks);
        --flash_ticks_;
        return w;
    }

    long long photon_count() const override { return photon_count_; }

protected:
    ses::WavepacketSimulation remake_simulation() const override { return make(); }
    const char* scene_name() const override { return "Harmonic trap"; }
    double default_camera_distance() const override { return 45.0; }

    std::string title_suffix() override {
        std::string s = strf("  w = %.2f au (T = %.1f au, E0 = %.3f Ha)",
                             kTrapOmega, 6.28318530717959 / kTrapOmega,
                             1.5 * kTrapOmega);
        if (decay_on_) {
            s += strf("  decay ON: photons %lld", photon_count_);
            if (!last_jump_.empty()) {
                s += strf(", last %s", last_jump_.c_str());
            }
        }
        if (!last_measure_.empty()) {
            s += strf("  measured %s", last_measure_.c_str());
        }
        s += "  5=excite D=decay E=measure";
        return s;
    }

    // Key E is deferred here (run_frame, psi quiescent, before stepping).
    void service_requests() override {
        if (!pending_measure_) {
            return;
        }
        pending_measure_ = false;
        engine_.project_psi();
        std::vector<double> pop(static_cast<std::size_t>(kNumTrapStates));
        for (int s = 0; s < kNumTrapStates; ++s) {
            pop[static_cast<std::size_t>(s)] =
                atom_.project_population(engine_, s);
        }
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        const int n = ses::sample_energy_eigenstate(pop, uniform(rng_));
        if (n >= 0) {
            atom_.collapse_onto(engine_, n);
            flush_collapse_error(n);
            write_display_texture();
            last_measure_ = strf("%s (E = %.3f Ha)", kTrapStates[n].name,
                                 atom_.state_energy(n));
        } else {
            // The deficit is the UNTRACKED bound ladder (N > 3), not a
            // continuum: collapse onto that complement by projecting the
            // tracked manifold out (a big coherent state, <N> ~ 8, lands
            // here most of the time -- honestly reported).
            project_manifold_out();
            last_measure_ = "outside tracked ladder (N > 3)";
        }
        title_dirty_ = true;
    }

    // Decay trials ride the real-time batch at title cadence, exactly the
    // atom's memoryless accumulated-dt scheme.
    void after_step_batch() override {
        if (!decay_on_ || atom_.channels().empty()) {
            return;
        }
        decay_accum_dt_ += pending_gpu_steps_ * sim_.dt();
        if (!gpu_title_due_) {
            return;
        }
        engine_.wait_async();  // the deposit needs the batch's memory visible
        engine_.project_psi();
        std::vector<double> rates(atom_.channels().size());
        for (std::size_t c = 0; c < atom_.channels().size(); ++c) {
            rates[c] = atom_.channels()[c].gamma_display *
                       atom_.project_population(engine_,
                                                atom_.channels()[c].from);
        }
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        const ses::ChannelPick pick = ses::pick_decay_channel(
            rates, decay_accum_dt_, uniform(rng_), uniform(rng_));
        decay_accum_dt_ = 0.0;
        if (pick.channel < 0) {
            return;
        }
        const ShellChannel& ch =
            atom_.channels()[static_cast<std::size_t>(pick.channel)];
        atom_.collapse_onto(engine_, ch.to);
        flush_collapse_error(ch.to);
        flash_ticks_ = kTrapFlashTicks;
        ++photon_count_;
        last_jump_ = strf("%s->%s", kTrapStates[ch.from].name,
                          kTrapStates[ch.to].name);
        std::fprintf(stderr, "trap decay: jump %s (photon #%lld, t=%.1f au)\n",
                     last_jump_.c_str(), photon_count_,
                     sim_.time() + gpu_time_);
        title_dirty_ = true;
    }

private:
    // Coherent state: sigma = 1/sqrt(2 w) (ground |psi|^2 width) -- no
    // breathing; the same width harmonic_dynamics_test pins (kSigmaGs).
    static ses::WavepacketSimulation make() {
        const ses::Grid1D axis{-kTrapBox, kTrapBox, 256};
        const ses::Grid3D grid{axis, axis, axis};
        const double sigma = 1.0 / std::sqrt(2.0 * kTrapOmega);
        return ses::WavepacketSimulation{ses::WavepacketSimulation::Config{
            grid,
            ses::harmonic_potential(grid, kTrapOmega, ses::Vec3d{}),
            ses::Vec3d{kCoherentOffset, 0.0, 0.0},
            ses::Vec3d{sigma, sigma, sigma},
            ses::Vec3d{},  // released at rest: x(t) = x0 cos(w t)
            0.04,          // dt
        }};
    }

    // Lazy manifold: radial solve (1D, instant) + channel table (factorized,
    // microseconds) + one norm-capturing synthesis pass per state + the
    // projection index -- first D/E only; the plain scene never pays.
    bool ensure_manifold() {
        if (!use_gpu_path()) {
            return false;
        }
        if (!atom_.radial_ready()) {
            const ses::RadialGrid rg{kTrapBox, 3999};
            std::vector<double> vr(static_cast<std::size_t>(rg.n));
            for (int i = 0; i < rg.n; ++i) {
                const double r = rg.r(i);
                vr[static_cast<std::size_t>(i)] =
                    0.5 * kTrapOmega * kTrapOmega * r * r;
            }
            atom_.solve_radial_manifold(rg, vr, kTrapStates, kNumTrapStates,
                                        kTrapLevels, kNumTrapLevels);
        }
        if (!proj_ready_) {
            const ses::RadialBinIndex bin_idx =
                ses::build_radial_bin_index(sim_.grid(), atom_.radial_grid());
            proj_ready_ = engine_.set_projection_index(
                bin_idx.sorted_cell, bin_idx.bin_off, atom_.radial_grid().n,
                atom_.radial_grid().h(), 3);  // l <= 3 in the N <= 3 ladder
            if (!proj_ready_) {
                std::fprintf(stderr, "trap: projection index setup failed -- "
                                     "measurement/decay disabled\n");
                return false;
            }
        }
        return atom_.prepare_manifold_cache(engine_, kTrapGammaDisplay);
    }

    void toggle_decay() {
        if (!decay_on_) {
            if (!ensure_manifold()) {
                return;
            }
            decay_accum_dt_ = 0.0;  // no hazard accrues while decay is off
        }
        decay_on_ = !decay_on_;
        // Flush residency ends with decay (mirrors the atom's policy).
        if (!decay_on_ && stepping_ == BaseStepping::RealTime) {
            engine_.release_relax_tables();
        }
    }

    void measure_energy_now() {
        if (!ensure_manifold()) {
            return;
        }
        pending_measure_ = true;
        stepping_ = BaseStepping::RealTime;
    }

    // Key 5: cycle prepared ladder eigenstates (static under the trap; the
    // Larmor readout drops to ~0, and D makes them cascade by QED jumps).
    void excite_next() {
        if (!ensure_manifold()) {
            return;
        }
        static constexpr int kCycle[] = {3, 7, 4, 12};  // 1p_z 2d_z2 2s 3p_z
        const int idx = kCycle[excite_cycle_++ % 4];
        atom_.collapse_onto(engine_, idx);
        flush_collapse_error(idx);
        cpu_is_truth_ = false;
        stepping_ = BaseStepping::RealTime;
        last_measure_.clear();
        title_dirty_ = true;
    }

    // Post-collapse eigenstate-error flush (same contracts as the atom:
    // fixed budget, ground gets the deep burst, tables resident while decay
    // is armed). No laser exists here, so no drive gating.
    void flush_collapse_error(int target) {
        if (!ensure_relax_tables()) {
            return;
        }
        engine_.relax_step(target == 0 ? kTrapFlushStepsGround
                                       : kTrapFlushSteps);
        if (!decay_on_) {
            engine_.release_relax_tables();
        }
    }

    // Collapse onto the untracked-ladder complement: subtract every tracked
    // amplitude, renormalize (the atom's continuum-verdict seam, N > 3 here).
    void project_manifold_out() {
        std::array<std::complex<double>, kNumTrapStates> amp{};
        double bound = 0.0;
        for (int s = 0; s < kNumTrapStates; ++s) {
            amp[static_cast<std::size_t>(s)] =
                atom_.project_state_amplitude(engine_, s);
            bound += std::norm(amp[static_cast<std::size_t>(s)]);
        }
        const double residual = engine_.norm_and_peak().sum - bound;
        if (residual <= 1e-4) {
            return;  // fp32 noise: nothing real to collapse onto
        }
        for (int s = 0; s < kNumTrapStates; ++s) {
            const std::complex<double> c = amp[static_cast<std::size_t>(s)];
            if (std::norm(c) < 1e-9) {
                continue;
            }
            const int buf = atom_.synth_transient(engine_, s);
            if (buf >= 0) {
                engine_.add_state_into_psi(buf, -c.real(), -c.imag());
                engine_.release_state(buf);
            }
        }
        const ses_vk::Engine::NormPeak np = engine_.norm_and_peak();
        if (np.sum > 1e-12) {
            engine_.scale(static_cast<float>(1.0 / std::sqrt(np.sum)));
        }
        cpu_is_truth_ = false;
        write_display_texture();
    }

    // Tracked ladder: radial solve, synthesis bookkeeping, channel table.
    ses_shell::AtomModel atom_;
    bool proj_ready_ = false;   // static projection index uploaded
    bool decay_on_ = false;
    bool pending_measure_ = false;  // Key E: serviced in run_frame
    double decay_accum_dt_ = 0.0;
    long long photon_count_ = 0;
    int excite_cycle_ = 0;
    int flash_ticks_ = 0;
    std::string last_jump_;
    std::string last_measure_;
};

}  // namespace ses_shell
