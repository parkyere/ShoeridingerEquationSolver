module;
#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <utility>
#include <vector>
export module ses.scenario.bloch1d_director;
export import ses.scenario.line1d_director;
import ses.bloch;
import ses.wavepacket;


// 1D optical lattice (Mathieu problem): V(x) = V0 sin^2(kL x) -- SMOOTH
// so the FFT split-operator keeps spectral accuracy (Kronig-Penney kinks
// would Gibbs-ring). Amber inset: exact E_n(q) from the tridiagonal
// central equation; cyan marker: live q on band 0. Tilt F implemented as
// the comoving gauge A(t) = -F t; Bloch oscillation T_B = G/F (F = 0:
// plain band-limited dispersion).


export namespace ses_shell {

constexpr double kBl1dKl = 1.0;               // lattice constant a = pi
constexpr int kBl1dPeriods = 26;              // integer periods in the box
constexpr int kBl1dPoints = 4096;
constexpr double kBl1dDt = 0.01;
constexpr double kBl1dV0 = 1.5;
constexpr double kBl1dV0Min = 0.0;
constexpr double kBl1dV0Max = 4.0;
constexpr double kBl1dFMax = 0.15;
constexpr double kBl1dF = 0.05;
constexpr double kBl1dSigma = 6.0;
constexpr double kBl1dRScale = 150.0;
constexpr double kBl1dEScale = 8.0;           // V display: Ha -> Bohr
constexpr int kBl1dBands = 3;
constexpr int kBl1dQSamples = 65;

class Bloch1DDirector final : public Line1DDirector, public BlochApi {
public:
    Bloch1DDirector()
        : Line1DDirector(box_grid(), lattice_potential(box_grid(), kBl1dV0),
                         kBl1dDt, kBl1dRScale, kBl1dEScale, 1e30) {
        rebuild_prop();
        rebuild_band_inset();
        fire();
    }

    BlochApi* bloch() override { return this; }

    // ---- BlochApi ----
    void set_depth(double v0) override {
        v0_ = std::clamp(v0, kBl1dV0Min, kBl1dV0Max);
        set_potential(lattice_potential(grid1d_, v0_));  // red curve too
        rebuild_prop();
        rebuild_band_inset();
        fire();
    }
    double depth() const override { return v0_; }
    void set_force(double f) override {
        force_ = std::clamp(f, 0.0, kBl1dFMax);
        rebuild_prop();
        fire();
    }
    double force() const override { return force_; }
    void refire() override { fire(); }
    double bloch_period() const override {
        return force_ > 0.0 ? 2.0 * kBl1dKl / force_ : 0.0;
    }
    // Mechanical quasimomentum q(t) = q0 + F t, folded to [-kL, kL).
    double quasimomentum() const override {
        const double g2 = 2.0 * kBl1dKl;
        double q = tilted_ ? tilted_->drift() : 0.0;
        q -= g2 * std::floor(q / g2 + 0.5);
        return q;
    }
    double mean_x() const override { return mean_x_; }
    double excursion() const override { return excursion_; }

    bool handle_key(char key) override {
        if (key == '2') {
            fire();
            return true;
        }
        return false;
    }

    double default_camera_azimuth() const override { return 0.25; }
    double default_camera_elevation() const override { return 0.22; }
    double default_camera_distance() const override { return 55.0; }

    // Base curves 0..3 + the band inset + the quasimomentum marker.
    int overlay_curve_count() const override { return 6; }
    OverlayCurve overlay_curve(int i) const override {
        if (i == 4) {  // band structure inset, amber
            return {band_curve_.data(),
                    static_cast<int>(band_curve_.size() / 3),
                    1.0f, 0.70f, 0.20f, 0.9f};
        }
        if (i == 5) {  // live q marker on band 0, cyan
            return {q_marker_.data(),
                    static_cast<int>(q_marker_.size() / 3),
                    0.35f, 0.85f, 1.0f, 1.0f};
        }
        return Line1DDirector::overlay_curve(i);
    }

protected:
    const char* scene_name() const override {
        return "1D crystal lattice (Bloch)";
    }

    std::string title_suffix() override {
        std::string s = strf(
            "  V0 = %.2f (E_R = %.2f)  F = %.3f  q = %+.2f kL  <x> = %+.1f "
            "(max |dx| %.1f)",
            v0_, 0.5 * kBl1dKl * kBl1dKl, force_,
            quasimomentum() / kBl1dKl, mean_x_, excursion_);
        if (force_ > 0.0) {
            s += strf("  T_Bloch = %.0f au", bloch_period());
        }
        s += "  keys: 2 refire";
        return s;
    }

    // The tilted propagator replaces the base split-operator wholesale.
    void step_batch(int n) override {
        tilted_->step(psi_, n);
        measure();
        rebuild_q_marker();
    }

    void after_reset() override { fire(); }

private:
    static ses::Grid1D box_grid() {
        const double half =
            0.5 * kBl1dPeriods * 3.14159265358979323846 / kBl1dKl;
        return ses::Grid1D{-half, half, kBl1dPoints};
    }

    static std::vector<double> lattice_potential(const ses::Grid1D& g,
                                                 double v0) {
        std::vector<double> v(static_cast<std::size_t>(g.n));
        for (int i = 0; i < g.n; ++i) {
            const double s = std::sin(kBl1dKl * g.coord(i));
            v[static_cast<std::size_t>(i)] = v0 * s * s;
        }
        return v;
    }

    void rebuild_prop() {
        tilted_ = std::make_unique<ses::TiltedSplitOperator1D>(
            grid1d_, lattice_potential(grid1d_, v0_), kBl1dDt, force_);
    }

    void fire() {
        tilted_->reset_time();
        sim_time_ = 0.0;
        pending_steps_ = 0;
        excursion_ = 0.0;
        // Broad packet at rest on a well minimum: ground band, q ~ 0.
        set_state(ses::gaussian_wavepacket(grid1d_, 0.0, kBl1dSigma, 0.0));
        measure();
        x0_ = mean_x_;
        rebuild_q_marker();
        note_dirty();
    }

    void measure() {
        double num = 0.0;
        double den = 0.0;
        for (int i = 0; i < grid1d_.n; ++i) {
            const double w = std::norm(psi_[i]);
            num += grid1d_.coord(i) * w;
            den += w;
        }
        mean_x_ = num / den;
        excursion_ = std::max(excursion_, std::abs(mean_x_ - x0_));
    }

    // Band inset: E_n(q) over the zone, drawn in the top-left corner in
    // world coordinates (bands chained left-right-left: the connectors at
    // the zone edges trace the gaps).
    void rebuild_band_inset() {
        band_curve_.clear();
        const double x_lo = grid1d_.xmin + 3.0;
        const double x_w = 24.0;
        e_ref_ = ses::lattice_bands(v0_, kBl1dKl, 0.0, kBl1dBands + 1);
        const double e_top =
            e_ref_[static_cast<std::size_t>(kBl1dBands)];
        y_scale_ = 13.0 / std::max(e_top, 1e-6);
        y_off_ = 12.0;
        for (int band = 0; band < kBl1dBands; ++band) {
            for (int s = 0; s < kBl1dQSamples; ++s) {
                const int idx =
                    band % 2 == 0 ? s : kBl1dQSamples - 1 - s;
                const double q =
                    -kBl1dKl +
                    2.0 * kBl1dKl * idx / (kBl1dQSamples - 1);
                const std::vector<double> e =
                    ses::lattice_bands(v0_, kBl1dKl, q, band + 1);
                band_curve_.push_back(static_cast<float>(
                    x_lo + x_w * (q + kBl1dKl) / (2.0 * kBl1dKl)));
                band_curve_.push_back(static_cast<float>(
                    y_off_ + y_scale_ * e[static_cast<std::size_t>(band)]));
                band_curve_.push_back(0.0f);
            }
        }
        rebuild_q_marker();
    }

    void rebuild_q_marker() {
        const double q = quasimomentum();
        const std::vector<double> e =
            ses::lattice_bands(v0_, kBl1dKl, q, 1);
        const double x_lo = grid1d_.xmin + 3.0;
        const double x_w = 24.0;
        const float mx = static_cast<float>(
            x_lo + x_w * (q + kBl1dKl) / (2.0 * kBl1dKl));
        const float my =
            static_cast<float>(y_off_ + y_scale_ * e[0]);
        q_marker_ = {mx, my - 1.0f, 0.0f, mx, my + 1.0f, 0.0f};
    }

    void note_dirty() {
        display_changed_ = true;
        title_dirty_ = true;
    }

    std::unique_ptr<ses::TiltedSplitOperator1D> tilted_;
    std::vector<float> band_curve_;
    std::vector<float> q_marker_;
    std::vector<double> e_ref_;
    double v0_ = kBl1dV0;
    double force_ = kBl1dF;
    double mean_x_ = 0.0;
    double x0_ = 0.0;
    double excursion_ = 0.0;
    double y_scale_ = 1.0;
    double y_off_ = 12.0;
};

}  // namespace ses_shell
