module;
#include <algorithm>
#include <complex>
#include <cstddef>
#include <string>
#include <vector>
export module ses.scenario.ptwell1d_director;
export import ses.scenario.line1d_director;
import ses.fft;
import ses.spectral;
import ses.wavepacket;


// Poschl-Teller V = -l(l+1)/(2 a^2) sech^2(x/a), integer l: reflectionless
// (KdV soliton potential); the equal-depth/area square well is not. Key W
// swaps the wells and relaunches the same packet; the HUD tracks R as the
// NEGATIVE-momentum probability (position gates would be polluted by the
// slow transmitted tail), frozen once the absorber has eaten most of the
// norm so absorbed flux cannot inflate the ratio.
// License physics: tests/solvable_wells_test.cpp (R_pt < 5e-3, R_sq > 3e-2).


export namespace ses_shell {

constexpr double kPt1dBox = 80.0;   // Bohr half-extent
constexpr int kPt1dPoints = 65536;
constexpr double kPt1dLambda = 2.0;
constexpr double kPt1dA = 2.0;
// The magic depth l(l+1)/(2 a^2) = 6/8 for l = 2, a = 2.
constexpr double kPt1dV0 = 0.75;
constexpr double kPt1dK0 = 0.5;     // E = 0.125 Ha
constexpr double kPt1dLaunchX = -30.0;
constexpr double kPt1dSigma = 5.0;
constexpr double kPt1dDt = 0.04;
constexpr double kPt1dAbsorb = 10.0;
constexpr double kPt1dRScale = 60.0;
constexpr double kPt1dEScale = 13.0;  // depth 0.75 dips ~10 Bohr below axis
constexpr double kPt1dYClamp = 1e30;

class PtWell1DDirector final : public Line1DDirector, public ReflectApi {
public:
    PtWell1DDirector()
        : Line1DDirector(ses::Grid1D{-kPt1dBox, kPt1dBox, kPt1dPoints},
                         ses::poschl_teller_potential(
                             ses::Grid1D{-kPt1dBox, kPt1dBox, kPt1dPoints},
                             kPt1dV0, kPt1dA),
                         kPt1dDt, kPt1dRScale, kPt1dEScale, kPt1dYClamp) {
        set_mask(ses::absorbing_mask(grid1d_, kPt1dAbsorb));
        launch();
    }

    ReflectApi* reflect() override { return this; }

    // ---- ReflectApi ----
    double reflected_max() const override { return r_max_; }
    bool square_well() const override { return square_; }
    void toggle_well() override {
        square_ = !square_;
        set_potential(square_
                          ? ses::barrier_potential(grid1d_, -kPt1dV0, -kPt1dA,
                                                   kPt1dA)
                          : ses::poschl_teller_potential(grid1d_, kPt1dV0,
                                                         kPt1dA));
        launch();  // same packet, fresh probes: a fair A/B comparison
    }

    bool handle_key(char key) override {
        if (key == 'W') {
            toggle_well();
            return true;
        }
        return false;
    }

    double default_camera_azimuth() const override { return 0.22; }
    double default_camera_elevation() const override { return 0.24; }
    double default_camera_distance() const override { return 185.0; }

protected:
    const char* scene_name() const override {
        return "1D reflectionless well (Poschl-Teller)";
    }
    int steps_per_tick() const override { return 2; }

    std::string title_suffix() override {
        return strf("  %s  E = %.3f Ha, depth %.2f Ha  R(k<0) = %.4f (max "
                    "%.4f)  [W] swap well",
                    square_ ? "SQUARE well (equal depth/area): edges reflect"
                            : "sech^2 lambda = 2: REFLECTIONLESS",
                    0.5 * kPt1dK0 * kPt1dK0, kPt1dV0, r_now_, r_max_);
    }

    // R at title cadence (a 64k FFT is ~ms; every batch would be waste).
    void after_batch() override {
        if (++probe_phase_ % 8 != 0) {
            return;
        }
        std::vector<std::complex<double>> phi = psi_.data();
        ses::fft(phi);
        const std::vector<double> kv = ses::wavenumbers(grid1d_);
        double neg = 0.0;
        double tot = 0.0;
        for (std::size_t j = 0; j < phi.size(); ++j) {
            const double w = std::norm(phi[j]);
            tot += w;
            if (kv[j] < 0.0) {
                neg += w;
            }
        }
        r_now_ = tot > 0.0 ? neg / tot : 0.0;
        // Record R only while the absorber has not touched the packet:
        // chopping a moving wave train (psi *= mask) is non-unitary and
        // MANUFACTURES spurious k < 0 sidebands (measured ~3% once the
        // transmitted front enters the ramp). The full reflection develops
        // by t ~ 120 au; anything within 15 Bohr of the ramp freezes the
        // record with the honest value already taken.
        const double guard = kPt1dBox - kPt1dAbsorb - 15.0;
        if (ses::probability_in_range(psi_, -guard, guard) > 1.0 - 1e-3) {
            r_max_ = std::max(r_max_, r_now_);
        }
        title_dirty_ = true;
    }

    void after_reset() override { launch(); }

private:
    void launch() {
        set_state(ses::gaussian_wavepacket(grid1d_, kPt1dLaunchX, kPt1dSigma,
                                           kPt1dK0));
        r_now_ = 0.0;
        r_max_ = 0.0;
        probe_phase_ = 0;
        title_dirty_ = true;
    }

    bool square_ = false;
    double r_now_ = 0.0;
    double r_max_ = 0.0;
    int probe_phase_ = 0;
};

}  // namespace ses_shell
