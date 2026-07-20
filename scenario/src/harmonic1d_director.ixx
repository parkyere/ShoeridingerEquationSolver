module;
#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>
export module ses.scenario.harmonic1d_director;
export import ses.scenario.line1d_director;
import ses.ladder;
import ses.mcwf1d;
import ses.wavepacket;


// Textbook 1D harmonic oscillator with ladder-operator controls. The ground
// state is the exact Gaussian (sigma = 1/sqrt(2 omega)); [U] applies a-dag,
// [D] applies a, [S] prepares a random coherent superposition, [2] resets
// to the current well's ground. Down from the ground state is refused by
// the operator itself (a|0> = 0: vanishing norm, psi untouched) -- physics,
// not a UI rule. Up is capped at the MEASURED grid representability
// ceiling (ses.ladder ho_level_cap): on this box the ceiling is the box
// itself -- the level's turning points must fit inside +-100 Bohr -- so
// the cap rises with omega (~1200 at w = 0.25, ~19000 at w = 4).
//
// The HUD classifies the state honestly with Var(H) -- no measurement, no
// basis bookkeeping: Var ~ 0 names the Fock level n; otherwise the state
// is a superposition and <N> = <H>/w - 1/2 is a real number. The well
// stiffness omega is live-adjustable as a sudden QUENCH: psi is KEPT, so
// the old state breathes in the new well (and the classifier duly reports
// a superposition); reset then lands in the new ground.


export namespace ses_shell {

constexpr double kHo1dOmega = 0.25;   // boot default; panel-adjustable
constexpr double kHo1dOmegaMin = 0.05;
// The dial spans two decades of stiffness; with the box-limited ceiling
// the cap now RISES with omega across the whole dial (Nyquist would only
// take over near w ~ k_max/x_max ~ 10, past the stop).
constexpr double kHo1dOmegaMax = 4.0;
// A 1D line is ~4 decades cheaper than the 256^3 volumes: the box is
// widened to raise the BOX-LIMITED level ceiling (turning point x_n =
// sqrt((2n+1)/w) must fit): +-100 holds n ~ 1200 at w = 0.25 and ~19000
// at w = 4 -- with the scaled Hermite chain (ses.ladder) the MEASURED
// ho_level_cap now reaches that box ceiling; nothing artificial remains
// below it. The line runs at 65536 points (2^16, h ~ 0.003 -- still
// 1/256th of one 256^3 volume, k_max ~ 1000, so Nyquist never binds on
// this dial). The huge k_max would murder the raw spectral chain, but
// eigenstates rung via the stable oracle path and superpositions via the
// streaming Fock-basis path -- neither cares about k_max.
constexpr double kHo1dBox = 100.0;    // Bohr half-extent
constexpr int kHo1dPoints = 65536;
constexpr double kHo1dDt = 0.04;
constexpr double kHo1dRScale = 18.0;  // radius = 18 |psi|^2 (~5 Bohr at n=0)
constexpr double kHo1dEScale = 0.8;   // V display: Ha -> Bohr height
// No display clamp: a clamped parabola reads as a FLAT-TOPPED (finite)
// well -- wrong physics on screen. The red curve honestly leaves the frame.
constexpr double kHo1dYClamp = 1e30;
// Var(H) below this reads as an eigenstate: grid eigenstates sit at
// ~1e-13, the closest superposition (adjacent levels, tiny weight eps has
// Var ~ eps^2 w^2) crosses it only below display relevance.
constexpr double kHo1dVarEigenTol = 1e-8;
constexpr int kHo1dRandomTop = 5;     // random superposition spans n = 0..5
// Schrodinger cat |a> + |-a>: lobe offset (alpha = x0 sqrt(w/2), <n> ~ 8
// at the boot omega) and the cavity photon-loss rate for the MCWF
// decoherence lens (each lost photon flips the cat parity).
constexpr double kHo1dCatX0 = 8.0;
constexpr double kHo1dKappa = 0.05;

class Harmonic1DDirector final : public Line1DDirector, public Ladder1dApi {
public:
    Harmonic1DDirector()
        : Line1DDirector(ses::Grid1D{-kHo1dBox, kHo1dBox, kHo1dPoints},
                         ses::harmonic_potential(
                             ses::Grid1D{-kHo1dBox, kHo1dBox, kHo1dPoints},
                             kHo1dOmega),
                         kHo1dDt, kHo1dRScale, kHo1dEScale, kHo1dYClamp),
          rng_(20260718u) {
        remeasure_caps();
        set_state(ground());
    }

    Ladder1dApi* ladder1d() override { return this; }

    // ---- Ladder1dApi ----
    int level() const override { return level_; }
    double level_energy() const override {
        return ses::mean_energy(psi_, potential_);
    }
    // Two regimes, two noise-free paths: a classified EIGENSTATE rungs via
    // the stable oracle-rebuilt path (representability ceiling applies); a
    // superposition rungs in the truncated Fock basis (ladder_fock: exact
    // coefficient action, capped by the tracked band). The raw spectral
    // chain -- whose noise cap collapses at this grid's k_max -- remains
    // only as the honest fallback for a state outside the Fock band.
    int max_level() const override {
        return level_ >= 0 ? cap_level_ : fock_top();
    }
    double omega() const override { return omega_; }

    bool ladder(bool up) override {
        const bool eigen = level_ >= 0;
        if (up) {
            const int cap = eigen ? cap_level_ : fock_top();
            const double mean_n =
                ses::mean_energy(psi_, potential_) / omega_ - 0.5;
            if (mean_n >= static_cast<double>(cap)) {
                note_ = strf("n = %d cap (%s)", cap,
                             eigen ? "grid band" : "fock band");
                title_dirty_ = true;
                return false;
            }
        }
        double norm2 = 0.0;
        if (eigen) {
            norm2 = ses::ladder_rung_stable(psi_, omega_, level_, up);
        } else {
            double residual = 0.0;
            ses::Field1D trial = psi_;
            norm2 = ses::ladder_fock(trial, omega_, up, fock_top(), &residual);
            if (residual < 1e-6) {
                if (norm2 >= 1e-6) {
                    psi_ = std::move(trial);
                }
            } else {
                // Outside the Fock band: the raw operator is the honest
                // (noise-amplifying) fallback.
                norm2 = up ? ses::ladder_raise(psi_, omega_)
                           : ses::ladder_lower(psi_, omega_);
            }
        }
        if (norm2 < 1e-6) {
            // ||a psi||^2 = <N> ~ 0: annihilation (psi untouched).
            note_ = "a|0> = 0: refused";
            title_dirty_ = true;
            return false;
        }
        note_.clear();
        classify();
        mark_display_dirty();
        title_dirty_ = true;
        return true;
    }

    void set_omega(double w) override {
        omega_ = std::clamp(w, kHo1dOmegaMin, kHo1dOmegaMax);
        // Sudden quench: swap the well under the LIVE psi (kept), retarget
        // reset at the new ground, re-measure both caps.
        set_potential(ses::harmonic_potential(grid1d_, omega_));
        set_reset_target(ground());
        remeasure_caps();
        note_ = "quench: psi kept";
        classify();
    }

    void random_superposition() override {
        // Complex-Gaussian amplitudes over n = 0..kHo1dRandomTop, each
        // basis state built DIRECTLY from the Hermite oracle (clean at any
        // grid k_max; the raw raise chain would already be noise-corrupted
        // here) -- a PURE coherent superposition (mixtures are not
        // representable in a wavefunction solver).
        ses::Field1D acc{grid1d_};
        std::normal_distribution<double> gauss;
        const int top = std::min(kHo1dRandomTop, fock_top());
        for (int n = 0; n <= top; ++n) {
            const ses::Field1D basis = ses::ho_eigenstate(grid1d_, omega_, n);
            const std::complex<double> c{gauss(rng_), gauss(rng_)};
            for (int i = 0; i < acc.size(); ++i) {
                acc[i] += c * basis[i];
            }
        }
        ses::normalize(acc);
        replace_state(std::move(acc));
        note_.clear();
        classify();
    }

    // ---- cat + photon-loss MCWF (the decoherence lens) ----
    // CONTRACT: tests/mcwf1d_test.cpp (parity flip, kappa bleed) +
    // --selftest-cat (scene-scale energy decay and jump count).
    void cat() override {
        const double sig = 1.0 / std::sqrt(2.0 * omega_);
        ses::Field1D a =
            ses::gaussian_wavepacket(grid1d_, kHo1dCatX0, sig, 0.0);
        const ses::Field1D b =
            ses::gaussian_wavepacket(grid1d_, -kHo1dCatX0, sig, 0.0);
        for (int i = 0; i < a.size(); ++i) {
            a[i] += b[i];
        }
        ses::normalize(a);
        replace_state(std::move(a));
        jumps_ = 0;
        note_ = "cat |a> + |-a>";
        classify();
    }
    void toggle_loss() override {
        kappa_ = kappa_ > 0.0 ? 0.0 : kHo1dKappa;
        note_ = kappa_ > 0.0 ? "photon loss ON" : "photon loss off";
        title_dirty_ = true;
    }
    bool loss_on() const override { return kappa_ > 0.0; }
    long long jump_count() const override { return jumps_; }

    // Lazy eigen-decomposition strip (0..100 eV = 3.675 Ha): recomputed
    // only when a MUTATION touched the weights (unitary evolution never
    // does). CONTRACT: tests/ho_spectrum_test.cpp (the core helper).
    int spectrum_count() override {
        ensure_spectrum();
        return static_cast<int>(spec_.size());
    }
    double spectrum_ev(int i) override {
        ensure_spectrum();
        return spec_[static_cast<std::size_t>(i)].first * 27.211386;
    }
    double spectrum_weight(int i) override {
        ensure_spectrum();
        return spec_[static_cast<std::size_t>(i)].second;
    }

    bool handle_key(char key) override {
        if (key == 'C') {
            cat();
            return true;
        }
        if (key == 'X') {
            toggle_loss();
            return true;
        }
        if (key == 'U') {
            ladder(true);
            return true;
        }
        if (key == 'D') {
            ladder(false);
            return true;
        }
        if (key == 'S') {
            random_superposition();
            return true;
        }
        if (key == '2') {
            reset_simulation();
            return true;
        }
        return false;
    }

    double default_camera_azimuth() const override { return 0.35; }
    double default_camera_elevation() const override { return 0.28; }
    double default_camera_distance() const override { return 55.0; }

protected:
    const char* scene_name() const override { return "1D harmonic oscillator"; }

    std::string title_suffix() override {
        const double e = ses::mean_energy(psi_, potential_);
        std::string s;
        if (level_ >= 0) {
            s = strf("  w = %.2f  n = %d (cap %d)  <H> = %.4f Ha "
                     "((n+1/2)w = %.4f)",
                     omega_, level_, cap_level_, e, (level_ + 0.5) * omega_);
        } else {
            s = strf("  w = %.2f  superposition  <N> = %.2f  <H> = %.4f Ha  "
                     "Var(H) = %.1e",
                     omega_, e / omega_ - 0.5, e,
                     ses::energy_variance(psi_, potential_));
        }
        if (kappa_ > 0.0) {
            s += strf("  kappa = %.2f  photons lost %lld", kappa_, jumps_);
        }
        if (!note_.empty()) {
            s += "  [" + note_ + "]";
        }
        s += "  keys: U up / D down / S random / C cat / X loss / 2 ground";
        return s;
    }

    // Loss on: interleave the MCWF photon-loss step (jump = parity flip,
    // no-jump = conditional kappa damping) with the unitary stride.
    void step_batch(int n) override {
        if (kappa_ <= 0.0) {
            Line1DDirector::step_batch(n);
            return;
        }
        ensure_damp();
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        bool flipped = false;
        for (int s = 0; s < n; ++s) {
            prop_->step(psi_);
            if (ses::photon_loss_step(psi_, omega_, potential_, kappa_,
                                      dt_, uni(rng_), *damp_)) {
                ++jumps_;
                flipped = true;
            }
        }
        if (flipped) {
            note_ = strf("photon #%lld: parity flip", jumps_);
            classify();
        }
    }

    void after_reset() override {
        note_.clear();
        classify();
    }

private:
    ses::Field1D ground() const {
        // sigma = 1/sqrt(2 omega): the exact HO ground state.
        return ses::gaussian_wavepacket(grid1d_, 0.0,
                                        1.0 / std::sqrt(2.0 * omega_), 0.0);
    }

    void ensure_spectrum() {
        if (!spec_dirty_) {
            return;
        }
        spec_dirty_ = false;
        spec_ = ses::ho1d_spectrum(psi_, omega_, 100.0 / 27.211386);
    }

    // The no-jump damping ITP rides omega and kappa; rebuild on change
    // (quench-safe: potential_ always matches omega_).
    void ensure_damp() {
        if (damp_ && damp_w_ == omega_ && damp_k_ == kappa_) {
            return;
        }
        damp_ = std::make_unique<ses::ImaginaryTimePropagator1D>(
            grid1d_, potential_, kappa_ * dt_ / (2.0 * omega_));
        damp_w_ = omega_;
        damp_k_ = kappa_;
    }

    // Honest state classification via Var(H): eigenstates name their n,
    // everything else is a superposition (level -1). Every mutation path
    // routes through here, so it doubles as the spectrum invalidation.
    void classify() {
        spec_dirty_ = true;
        if (ses::energy_variance(psi_, potential_) < kHo1dVarEigenTol) {
            const double e = ses::mean_energy(psi_, potential_);
            level_ = static_cast<int>(std::lround(e / omega_ - 0.5));
        } else {
            level_ = -1;
        }
    }

    // ho_level_cap sweeps the whole scaled chain (up to the box ceiling,
    // ~1200 levels at w = 0.25 and ~19000 at w = 4 on this grid), so each
    // distinct omega is measured ONCE and memoized -- revisiting a slider
    // value costs nothing.
    void remeasure_caps() {
        for (const auto& [w, cap] : cap_memo_) {
            if (w == omega_) {
                cap_level_ = cap;
                return;
            }
        }
        cap_level_ = ses::ho_level_cap(grid1d_, omega_);
        cap_memo_.emplace_back(omega_, cap_level_);
    }
    // Fock band top for superposition laddering: the representability
    // ceiling is the ONLY cap (an up-rung targets fock_top() + 1); the
    // streaming ladder_fock scans just the levels the state occupies, so a
    // deep band top costs nothing for low superpositions.
    int fock_top() const { return cap_level_ - 1; }

    double omega_ = kHo1dOmega;
    std::vector<std::pair<double, int>> cap_memo_;  // measured cap per omega
    int cap_level_ = 0;  // stable rungs: grid representability ceiling
    int level_ = 0;
    std::string note_;
    std::mt19937 rng_;
    // Lazy spectrum cache (dirty on every state mutation).
    std::vector<std::pair<double, double>> spec_;
    bool spec_dirty_ = true;
    // Photon-loss MCWF state (the cat decoherence lens).
    double kappa_ = 0.0;
    std::unique_ptr<ses::ImaginaryTimePropagator1D> damp_;
    double damp_w_ = -1.0;
    double damp_k_ = -1.0;
    long long jumps_ = 0;
};

}  // namespace ses_shell
