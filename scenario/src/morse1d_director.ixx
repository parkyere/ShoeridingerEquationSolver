module;
#include <cstddef>
#include <cmath>
#include <complex>
#include <string>
#include <utility>
#include <vector>
export module ses.scenario.morse1d_director;
export import ses.scenario.line1d_director;
import ses.spectrum1d;


// Morse well: E_n = w0 (n + 1/2) - (alpha^2/2)(n + 1/2)^2, w0 = alpha sqrt(2 d);
// finite bound set below d. [U]/[D] jump eigenstates (closed form locked in
// tests/solvable_wells_test.cpp), [S] (n, n+1) pair beat T = 2pi/gap, [2] ground.


export namespace ses_shell {

constexpr double kMo1dBox = 100.0;  // Bohr half-extent
constexpr int kMo1dPoints = 65536;
constexpr double kMo1dD = 0.3;      // dissociation limit (Ha)
constexpr double kMo1dAlpha = 0.12;
constexpr double kMo1dX0 = -30.0;   // well minimum
constexpr double kMo1dDt = 0.04;
constexpr double kMo1dRScale = 40.0;
constexpr double kMo1dEScale = 30.0;  // d = 0.3 -> plateau ~9 Bohr high
constexpr double kMo1dYClamp = 1e30;

class Morse1DDirector final : public Line1DDirector, public MorseApi {
public:
    Morse1DDirector()
        : Line1DDirector(ses::Grid1D{-kMo1dBox, kMo1dBox, kMo1dPoints},
                         ses::morse_potential(
                             ses::Grid1D{-kMo1dBox, kMo1dBox, kMo1dPoints},
                             kMo1dD, kMo1dAlpha, kMo1dX0),
                         kMo1dDt, kMo1dRScale, kMo1dEScale, kMo1dYClamp) {
        // Solve past the bound set; keep only genuinely bound levels (the
        // rest are box-discretized continuum).
        std::vector<ses::Bound1D> s =
            ses::bound_states_1d(grid1d_, potential_, 8);
        for (ses::Bound1D& b : s) {
            if (b.energy < kMo1dD - 2e-3) {
                bound_.push_back(std::move(b));
            }
        }
        set_state(clone(0));
    }

    MorseApi* morse() override { return this; }

    // ---- MorseApi ----
    int level() const override { return level_; }
    double level_energy() const override {
        return ses::mean_energy(psi_, potential_);
    }
    int bound_count() const override { return static_cast<int>(bound_.size()); }
    bool jump(bool up) override {
        // From the pair superposition, jump re-prepares from its base level.
        const int from = level_ >= 0 ? level_ : base_;
        const int target = from + (up ? 1 : -1);
        if (target < 0 || target >= bound_count()) {
            note_ = up ? "top bound level (dissociation above)"
                       : "already the ground state";
            title_dirty_ = true;
            return false;
        }
        level_ = target;
        base_ = target;
        replace_state(clone(target));
        set_reset_target(clone(0));
        note_.clear();
        title_dirty_ = true;
        return true;
    }

    bool handle_key(char key) override {
        if (key == 'U') {
            jump(true);
            return true;
        }
        if (key == 'D') {
            jump(false);
            return true;
        }
        if (key == 'S') {
            pair_superposition();
            return true;
        }
        if (key == '2') {
            level_ = 0;
            base_ = 0;
            note_.clear();
            replace_state(clone(0));
            return true;
        }
        return false;
    }

    double default_camera_azimuth() const override { return 0.3; }
    double default_camera_elevation() const override { return 0.25; }
    double default_camera_distance() const override { return 120.0; }

protected:
    const char* scene_name() const override {
        return "1D Morse well (anharmonic ladder)";
    }

    std::string title_suffix() override {
        std::string s;
        if (level_ >= 0) {
            const double e = bound_[static_cast<std::size_t>(level_)].energy;
            s = strf("  n = %d/%d  E = %.4f Ha (D = %.2f)", level_,
                     bound_count() - 1, e, kMo1dD);
            if (level_ + 1 < bound_count()) {
                s += strf("  gap up = %.4f",
                          bound_[static_cast<std::size_t>(level_ + 1)].energy - e);
            }
        } else {
            const double gap =
                bound_[static_cast<std::size_t>(base_ + 1)].energy -
                bound_[static_cast<std::size_t>(base_)].energy;
            s = strf("  pair (%d, %d)  beat T = 2pi/gap = %.0f au (slows "
                     "up the ladder)",
                     base_, base_ + 1, 2.0 * 3.14159265358979 / gap);
        }
        if (!note_.empty()) {
            s += "  [" + note_ + "]";
        }
        s += "  keys: U up / D down / S pair / 2 ground";
        return s;
    }

    void after_reset() override {
        level_ = 0;
        base_ = 0;
        note_.clear();
    }

private:
    ses::Field1D clone(int k) const {
        return bound_[static_cast<std::size_t>(k)].psi;
    }

    void pair_superposition() {
        if (base_ + 1 >= bound_count()) {
            note_ = "no level above for a pair";
            title_dirty_ = true;
            return;
        }
        ses::Field1D psi{grid1d_};
        const ses::Field1D& a = bound_[static_cast<std::size_t>(base_)].psi;
        const ses::Field1D& b = bound_[static_cast<std::size_t>(base_ + 1)].psi;
        for (int i = 0; i < grid1d_.n; ++i) {
            psi[i] = a[i] + b[i];
        }
        ses::normalize(psi);
        level_ = -1;
        note_.clear();
        replace_state(std::move(psi));
    }

    std::vector<ses::Bound1D> bound_;
    int level_ = 0;
    int base_ = 0;
    std::string note_;
};

}  // namespace ses_shell
