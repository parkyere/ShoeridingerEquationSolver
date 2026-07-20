module;
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>
export module ses.scenario.anderson1d_director;
export import ses.scenario.line1d_director;
import ses.wavepacket;
import ses.observables;
import ses.field;


// Anderson localization in 1D: a packet with energy ABOVE every barrier
// (a classical particle sails through) is STOPPED by coherent multiple
// scattering off a random landscape -- in 1D every eigenstate is
// exponentially localized for any disorder. Sharp scatterers on purpose
// (sigma < lambda): smooth bumps are transparent at k (the corral fence
// lesson), sharp ones backscatter. Clean (W = 0) contrast: ballistic
// flight. CONTRACT: tests/anderson1d_test.cpp.


export namespace ses_shell {

// SPECKLE disorder (the cold-atom realization, Billy et al. 2008): dense
// overlapping bumps -> a smooth random field with correlation length ~
// sigma, every correlation cell a scatterer. Strength W ~ E (the standard
// Anderson regime): sub-E-everywhere fields proved far too transparent
// (measured <= 36% blocking over 100 Bohr at every sub-E tuning) -- the
// localization length simply exceeds any reasonable stage.
constexpr double kAn1dSpacing = 0.6;    // speckle grain spacing (Bohr)
constexpr double kAn1dBumpSigma = 0.3;  // correlation length
constexpr double kAn1dK0 = 1.2;         // E = 0.72 Ha
constexpr double kAn1dW = 1.2;          // grain amplitude range [-W, W]

// Random landscape: Gaussian bumps at every lattice site, amplitudes
// uniform in [-w, w] from the SEEDED mt19937 (deterministic per seed).
inline std::vector<double> anderson_potential(const ses::Grid1D& g, double w,
                                              unsigned seed) {
    std::vector<double> v(static_cast<std::size_t>(g.n), 0.0);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> amp(-w, w);
    const double margin = 2.0;  // no scatterer inside the box lip
    const int s_lo = static_cast<int>(
        std::ceil((g.xmin + margin) / kAn1dSpacing));
    const int s_hi = static_cast<int>(
        std::floor((g.xmax - margin) / kAn1dSpacing));
    for (int s = s_lo; s <= s_hi; ++s) {
        const double xs = s * kAn1dSpacing;
        const double a = amp(rng);
        for (int i = 0; i < g.n; ++i) {
            const double dx = g.coord(i) - xs;
            if (std::abs(dx) > 5.0 * kAn1dBumpSigma) {
                continue;
            }
            v[static_cast<std::size_t>(i)] +=
                a * std::exp(-dx * dx /
                             (2.0 * kAn1dBumpSigma * kAn1dBumpSigma));
        }
    }
    return v;
}

constexpr int kAn1dPoints = 4096;
constexpr double kAn1dHalf = 60.0;
constexpr double kAn1dDt = 0.01;
constexpr double kAn1dLaunchX = -45.0;
constexpr double kAn1dSigma0 = 2.0;
constexpr double kAn1dWMax = 2.0;
constexpr double kAn1dCapW0 = 4.0;
constexpr double kAn1dCapWidth = 6.0;
constexpr double kAn1dRScale = 150.0;
constexpr double kAn1dEScale = 8.0;

class Anderson1DDirector final : public Line1DDirector, public AndersonApi {
public:
    Anderson1DDirector()
        : Line1DDirector(wire_grid(),
                         anderson_potential(wire_grid(), kAn1dW, 1),
                         kAn1dDt, kAn1dRScale, kAn1dEScale, 1e30) {
        build_cap();
        fire();
    }

    AndersonApi* anderson() override { return this; }

    // ---- AndersonApi ----
    void set_disorder(double w) override {
        w_ = std::clamp(w, 0.0, kAn1dWMax);
        set_potential(anderson_potential(grid1d_, w_, seed_));
        fire();
    }
    double disorder() const override { return w_; }
    void reroll() override {
        ++seed_;
        set_potential(anderson_potential(grid1d_, w_, seed_));
        fire();
    }
    void refire() override { fire(); }
    double transmitted() const override { return transmitted_; }
    double survived() const override { return survived_; }

    bool handle_key(char key) override {
        if (key == '2') {
            fire();
            return true;
        }
        if (key == '5') {
            reroll();
            return true;
        }
        return false;
    }

    double default_camera_azimuth() const override { return 0.25; }
    double default_camera_elevation() const override { return 0.22; }
    double default_camera_distance() const override { return 80.0; }

protected:
    const char* scene_name() const override {
        return "Anderson localization (1D speckle wire)";
    }
    int steps_per_tick() const override { return 8; }

    std::string title_suffix() override {
        return strf(
            "  W = %.2f (E = %.2f)  seed %u  transmitted %.1f%%  "
            "on stage %.1f%%  keys: 2 refire / 5 new landscape",
            w_, 0.5 * kAn1dK0 * kAn1dK0, seed_, 100.0 * transmitted_,
            100.0 * survived_);
    }

    // Step + right-cap transmission tally (the conductance readout: the
    // test's exact metric) + the CAP frame; no renormalization -- the
    // norm IS the on-stage electron.
    void step_batch(int n) override {
        const double h = grid1d_.spacing();
        for (int s = 0; s < n; ++s) {
            prop_->step(psi_, 1);
            for (int i = 0; i < grid1d_.n; ++i) {
                const double c = cap_[static_cast<std::size_t>(i)];
                if (c < 1.0 && grid1d_.coord(i) > 0.0) {
                    transmitted_ += std::norm(psi_[i]) * (1.0 - c * c) * h;
                }
                psi_[i] *= c;
            }
        }
        survived_ = ses::norm_sq(psi_);
    }

    void after_reset() override { fire(); }

private:
    static ses::Grid1D wire_grid() {
        return ses::Grid1D{-kAn1dHalf, kAn1dHalf, kAn1dPoints};
    }

    void build_cap() {
        cap_.assign(static_cast<std::size_t>(grid1d_.n), 1.0);
        for (int i = 0; i < grid1d_.n; ++i) {
            const double d = std::min(grid1d_.coord(i) - grid1d_.xmin,
                                      grid1d_.xmax - grid1d_.coord(i));
            if (d < kAn1dCapWidth) {
                const double t = 1.0 - d / kAn1dCapWidth;
                cap_[static_cast<std::size_t>(i)] =
                    std::exp(-kAn1dCapW0 * t * t * kAn1dDt);
            }
        }
    }

    void fire() {
        transmitted_ = 0.0;
        survived_ = 1.0;
        sim_time_ = 0.0;
        pending_steps_ = 0;
        set_state(ses::gaussian_wavepacket(grid1d_, kAn1dLaunchX,
                                           kAn1dSigma0, kAn1dK0));
        display_changed_ = true;
        title_dirty_ = true;
    }

    std::vector<double> cap_;
    double w_ = kAn1dW;
    unsigned seed_ = 1;
    double transmitted_ = 0.0;
    double survived_ = 1.0;
};

}  // namespace ses_shell
