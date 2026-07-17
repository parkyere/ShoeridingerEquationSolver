module;
#include <algorithm>
#include <string>
export module ses.scenario.tunneling1d_director;
export import ses.scenario.line1d_director;
import ses.wavepacket;


// Textbook 1D tunneling: the same physics as the 3D tunneling scene (same
// V0, slab, launch, k0 -- E = k^2/2 = 0.125 < V0 = 0.25, classically
// forbidden) reduced to its textbook axis. The barrier is VISIBLE as the
// red potential profile (a rectangle), the packet as the white phasor
// curve: incident + reflected interference ripples left of the wall, the
// evanescent decay inside it, and the small transmitted curl escaping
// right. The boundary absorber swallows outgoing flux; T is reported
// against the initial unit norm (probability_in_range is absolute).


export namespace ses_shell {

constexpr double kTun1dBox = 80.0;   // Bohr half-extent
// 65536 points (2^16, h ~ 0.0024): a 1D line is ~4 decades cheaper than
// the 3D volumes, so oversample generously -- the phasor curve resolves
// the incident/reflected interference ripples and the evanescent decay
// smoothly. No ladder here, so the huge k_max costs nothing.
constexpr int kTun1dPoints = 65536;
constexpr double kTun1dV0 = 0.25;    // Ha
constexpr double kTun1dXLo = 0.0;    // slab [0, 5): kappa = 0.5, ~2-Bohr glow
constexpr double kTun1dXHi = 5.0;
constexpr double kTun1dK0 = 0.5;     // mean E = 0.125 Ha < V0 (forbidden)
constexpr double kTun1dLaunchX = -30.0;
constexpr double kTun1dSigma = 5.0;
constexpr double kTun1dDt = 0.04;
constexpr double kTun1dAbsorb = 10.0;  // cos^2 wall layer (Bohr)
constexpr double kTun1dRScale = 60.0;  // radius = 60 |psi|^2 (~5 Bohr peak)
constexpr double kTun1dEScale = 40.0;  // V display: barrier reads 10 tall
constexpr double kTun1dYClamp = 12.0;

class Tunneling1DDirector final : public Line1DDirector, public TunnelApi {
public:
    Tunneling1DDirector()
        : Line1DDirector(ses::Grid1D{-kTun1dBox, kTun1dBox, kTun1dPoints},
                         ses::barrier_potential(
                             ses::Grid1D{-kTun1dBox, kTun1dBox, kTun1dPoints},
                             kTun1dV0, kTun1dXLo, kTun1dXHi),
                         kTun1dDt, kTun1dRScale, kTun1dEScale, kTun1dYClamp) {
        set_mask(ses::absorbing_mask(grid1d_, kTun1dAbsorb));
        set_state(ses::gaussian_wavepacket(grid1d_, kTun1dLaunchX, kTun1dSigma,
                                           kTun1dK0));
    }

    TunnelApi* tunnel() override { return this; }
    double transmitted_max() const override { return t_max_; }

    // Near side-on, like the 3D scene: packet left, wall a thin red gate,
    // transmission right; a slight angle keeps the phasor twist readable.
    double default_camera_azimuth() const override { return 0.22; }
    double default_camera_elevation() const override { return 0.24; }
    double default_camera_distance() const override { return 185.0; }

protected:
    const char* scene_name() const override { return "1D quantum tunneling"; }
    // The packet crawls at v = k0 = 0.5 Bohr/au: two steps per tick keeps
    // the approach ~12 s of wall clock at scale 1.
    int steps_per_tick() const override { return 2; }

    std::string title_suffix() override {
        return strf("  V0 = %.2f Ha, E = %.3f Ha (forbidden)  P(x<%.0f) %.3f | "
                    "P(x>%.0f) %.3f (max T %.3f)",
                    kTun1dV0, 0.5 * kTun1dK0 * kTun1dK0, kTun1dXLo, p_left_,
                    kTun1dXHi, p_right_, t_max_);
    }

    // 1024 points: the probe is microseconds, so track T every batch.
    void after_batch() override {
        p_left_ = ses::probability_in_range(psi_, grid1d_.xmin, kTun1dXLo);
        p_right_ = ses::probability_in_range(psi_, kTun1dXHi, grid1d_.xmax);
        t_max_ = std::max(t_max_, p_right_);
    }

    void after_reset() override {
        p_left_ = 0.0;
        p_right_ = 0.0;
        t_max_ = 0.0;
    }

private:
    double p_left_ = 0.0;
    double p_right_ = 0.0;
    double t_max_ = 0.0;
};

}  // namespace ses_shell
