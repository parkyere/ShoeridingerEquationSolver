module;
#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <string>
export module ses.scenario.harmonic1d_director;
export import ses.scenario.line1d_director;
import ses.ladder;
import ses.wavepacket;


// Textbook 1D harmonic oscillator with ladder-operator controls. The ground
// state is the exact Gaussian (sigma = 1/sqrt(2 omega)); [U] applies a-dag,
// [D] applies a, [S] prepares a random coherent superposition, [2] resets
// to the current well's ground. Down from the ground state is refused by
// the operator itself (a|0> = 0: vanishing norm, psi untouched) -- physics,
// not a UI rule. Up is capped ADAPTIVELY at ses.ladder's ladder_cap: the
// FFT ladder amplifies the k_max round-off floor by k_max/sqrt(2w) per
// raise, so a softer well caps lower and a stiffer one higher.
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
// Widened past the old 1.0 to span the MEASURED clean-cap peak (~16 at
// w ~ 1, where the a-dag derivative and x gains balance) and its fall-off
// (~13 at w = 4): cranking the well past the peak now visibly LOWERS the
// ladder cap, the payoff of the empirical ladder_cap probe.
constexpr double kHo1dOmegaMax = 4.0;
constexpr double kHo1dBox = 20.0;     // Bohr half-extent
// 256 points: k_max ~ 20 vs the low-n band -- see tests/ladder_test.cpp
// (grid Nyquist matched to the physics band keeps the chain clean).
constexpr int kHo1dPoints = 256;
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
    // Two regimes, two caps: a classified EIGENSTATE rungs via the stable
    // oracle-rebuilt path (noise resets every rung -> the grid's
    // representability ceiling applies); a superposition must take the raw
    // spectral operator (no single oracle to rebuild from), so the raw
    // chain's noise cap applies -- superpositions live at low n anyway.
    int max_level() const override {
        return level_ >= 0 ? cap_level_ : cap_raw_;
    }
    double omega() const override { return omega_; }

    bool ladder(bool up) override {
        const bool eigen = level_ >= 0;
        if (up) {
            const int cap = eigen ? cap_level_ : cap_raw_;
            const double mean_n =
                ses::mean_energy(psi_, potential_) / omega_ - 0.5;
            if (mean_n >= static_cast<double>(cap)) {
                note_ = strf("n = %d cap (%s)", cap,
                             eigen ? "grid band" : "raw-chain noise");
                title_dirty_ = true;
                return false;
            }
            if (eigen) {
                ses::ladder_rung_stable(psi_, omega_, level_, true);
            } else {
                ses::ladder_raise(psi_, omega_);
            }
        } else {
            // ||a psi||^2 = <N>; ~0 means annihilation (psi untouched).
            const double norm2 =
                eigen ? ses::ladder_rung_stable(psi_, omega_, level_, false)
                      : ses::ladder_lower(psi_, omega_);
            if (norm2 < 1e-6) {
                note_ = "a|0> = 0: refused";
                title_dirty_ = true;
                return false;
            }
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
        // basis state built by the clean ladder chain from the ground --
        // a PURE coherent superposition (mixtures are not representable).
        ses::Field1D acc{grid1d_};
        ses::Field1D basis = ground();
        std::normal_distribution<double> gauss;
        const int top = std::min(kHo1dRandomTop, cap_raw_);
        for (int n = 0; n <= top; ++n) {
            if (n > 0) {
                ses::ladder_raise(basis, omega_);
            }
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

    bool handle_key(char key) override {
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
        if (!note_.empty()) {
            s += "  [" + note_ + "]";
        }
        s += "  keys: U up / D down / S random / 2 ground";
        return s;
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

    // Honest state classification via Var(H): eigenstates name their n,
    // everything else is a superposition (level -1).
    void classify() {
        if (ses::energy_variance(psi_, potential_) < kHo1dVarEigenTol) {
            const double e = ses::mean_energy(psi_, potential_);
            level_ = static_cast<int>(std::lround(e / omega_ - 0.5));
        } else {
            level_ = -1;
        }
    }

    void remeasure_caps() {
        cap_level_ = ses::ho_level_cap(grid1d_, omega_);
        cap_raw_ = ses::ladder_cap(grid1d_, omega_);
    }

    double omega_ = kHo1dOmega;
    int cap_level_ = 0;  // stable rungs: grid representability ceiling
    int cap_raw_ = 0;    // raw spectral chain (superpositions): noise cap
    int level_ = 0;
    std::string note_;
    std::mt19937 rng_;
};

}  // namespace ses_shell
