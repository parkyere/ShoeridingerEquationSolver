module;
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <utility>
#include <vector>
export module ses.scenario.qdot2d_director;
export import ses.scenario.lattice2d_director;
import ses.parallel;
import ses.heightfield;


// 2D quantum dot: parabolic confinement 1/2 w0^2 r^2 with an optional
// uniform B along z -- the FOCK-DARWIN problem. The ground relaxed by the
// lattice imaginary time (the link phases ride along, so the B != 0
// ground is reachable) must land at E = Omega = sqrt(w0^2 + B^2/4), the
// live HUD comparison. [F] displaces the relaxed ground: at B = 0 it
// swings as a coherent state at w0; with B it traces the two-frequency
// Fock-Darwin rosette (omega_pm = Omega -+ B/2), breadcrumbed white.


export namespace ses_shell {

constexpr double kQd2dBox = 20.0;
constexpr int kQd2dN = 512;
constexpr int kQd2dNz = 4;
constexpr double kQd2dZHalf = 2.0;
// dt rides h^2 (the landau rule): 512^2 at dt = 0.01 biases the relax
// fixed point +0.22 Ha off the Fock-Darwin ground (Trotter artifact, the
// benzene disease); 0.0025 restores the 256^2 bond angle.
constexpr double kQd2dDt = 0.0025;
constexpr double kQd2dW0 = 0.5;
constexpr double kQd2dW0Min = 0.2;
constexpr double kQd2dW0Max = 1.0;
constexpr double kQd2dB = 0.6;
constexpr double kQd2dBMax = 1.2;
constexpr double kQd2dDisplace = 4.0;
constexpr double kQd2dConvTol = 1e-8;
constexpr int kQd2dTrailCap = 900;
// IBM-style STM height surface (the corral rule): z = |psi|^2, peak SNAP
// then 0.98-decay -- the base's decay-only peak_ boots ~100x too high and
// blacked the cloud out for seconds.
constexpr double kQd2dSurfH = 6.0;
constexpr int kQd2dMeshStride = 1;  // 512^2 physics = 512^2 display mesh

class Qdot2DDirector final : public Lattice2DDirectorBase, public QdotApi {
public:
    Qdot2DDirector()
        : Lattice2DDirectorBase(kQd2dBox, kQd2dN, kQd2dNz, kQd2dZHalf) {
        rebuild_prop();
        relax_ground();
    }

    QdotApi* qdot() override { return this; }

    // ---- QdotApi ----
    void set_omega0(double w0) override {
        w0_ = std::clamp(w0, kQd2dW0Min, kQd2dW0Max);
        rebuild_prop();
        relax_ground();
    }
    double omega0() const override { return w0_; }
    void set_field(double b) override {
        b_ = std::clamp(b, 0.0, kQd2dBMax);
        prop_->set_uniform_field(b_);
        relax_ground();
    }
    double field() const override { return b_; }
    void relax_ground() override {
        ses::parallel_for(kQd2dN, [&](int j) {
            const double y = phys_grid_.y.coord(j);
            for (int i = 0; i < kQd2dN; ++i) {
                const double x = phys_grid_.x.coord(i);
                psi_(i, j, 0) = std::exp(-((x - 1.0) * (x - 1.0) + y * y) /
                                         (4.0 * 3.0 * 3.0));
            }
        });
        ses::normalize(psi_);
        relaxing_ = true;
        conv_streak_ = 0;
        last_e_ = 1e30;
        trail_.clear();
        track_peak();
        mark_fired();
    }
    bool relaxing() const override { return relaxing_; }
    double energy_meas() const override { return prop_->energy(psi_); }
    double energy_pred() const override {
        return std::sqrt(w0_ * w0_ + 0.25 * b_ * b_);
    }
    void fire_displaced() override {
        // Rigid shift of the current (relaxed) state: a coherent kick.
        relaxing_ = false;
        const int di = static_cast<int>(kQd2dDisplace /
                                        phys_grid_.x.spacing() +
                                        0.5);
        ses::Field3D shifted{phys_grid_};
        for (int j = 0; j < kQd2dN; ++j) {
            for (int i = 0; i < kQd2dN; ++i) {
                const int src = i - di;
                shifted(i, j, 0) =
                    src >= 0 && src < kQd2dN ? psi_(src, j, 0) : 0.0;
            }
        }
        ses::normalize(shifted);
        psi_ = std::move(shifted);
        trail_.clear();
        track_peak();
        mark_fired();
    }

    void reset_simulation() override { relax_ground(); }

    bool handle_key(char key) override {
        if (key == '2') {
            relax_ground();
            return true;
        }
        if (key == 'F') {
            fire_displaced();
            return true;
        }
        return false;
    }

    double sim_dt() const override { return kQd2dDt; }
    // x4 the base's 8: dt shrank x4 for the 512^2 bond angle, so the
    // rosette keeps its visual pace.
    int steps_per_tick() const override { return 32; }

    // ---- STM-style surface display (mesh path; cloud off) ----
    bool cloud() const override { return false; }
    const ses::Mesh& mesh() const override { return hf_.mesh; }
    const std::vector<ses::Rgb>& colors() const override {
        return hf_.colors;
    }
    bool take_mesh_dirty() override {
        return std::exchange(mesh_dirty_, false);
    }
    // Tilted boot view so the height relief reads (face-on would hide it).
    double default_camera_azimuth() const override { return 0.35; }
    double default_camera_elevation() const override { return 0.95; }
    double default_camera_distance() const override { return 75.0; }

protected:
    // Heightfield surface instead of the volume slab (corral peak rule).
    void rebuild_display() override {
        double cur = 0.0;
        for (int j = 0; j < kQd2dN; ++j) {
            for (int i = 0; i < kQd2dN; ++i) {
                cur = std::max(cur, std::norm(psi_(i, j, 0)));
            }
        }
        disp_peak_ = disp_peak_ <= 0.0 ? cur
                                       : std::max(cur, 0.98 * disp_peak_);
        hf_ = ses::heightfield_surface(psi_, kQd2dSurfH, disp_peak_,
                                       kQd2dMeshStride);
        mesh_dirty_ = true;
    }

public:

    std::string title_text() override {
        std::string s = strf(
            "2D quantum dot (Fock-Darwin)  |  t = %.1f au  w0 = %.2f  "
            "B = %.2f  E = %.4f (Omega = %.4f)",
            sim_time_, w0_, b_, energy_meas(), energy_pred());
        if (relaxing_) {
            s += "  [relaxing...]";
        }
        s += "  keys: 2 relax ground / F displace";
        return s;
    }

    // The measured <r>(t) breadcrumb trail (the rosette).
    int overlay_curve_count() const override { return 1; }
    OverlayCurve overlay_curve(int /*i*/) const override {
        return {trail_.data(), static_cast<int>(trail_.size() / 3),
                1.0f, 1.0f, 1.0f, 0.9f};
    }

protected:
    void do_steps(int n) override {
        if (relaxing_) {
            prop_->relax(psi_, 4 * n);
            const double e = prop_->energy(psi_);
            if (std::abs(e - last_e_) < kQd2dConvTol * std::max(1.0, e)) {
                if (++conv_streak_ >= 3) {
                    relaxing_ = false;
                    title_dirty_ = true;
                }
            } else {
                conv_streak_ = 0;
            }
            last_e_ = e;
            track_peak();
            return;
        }
        int left = n;
        while (left > 0) {
            const int chunk = std::min(left, 8);
            prop_->step(psi_, chunk);
            sim_time_ += chunk * kQd2dDt;
            left -= chunk;
            push_trail();
        }
        track_peak();
    }

private:
    void rebuild_prop() {
        std::vector<double> v(
            static_cast<std::size_t>(phys_grid_.size()));
        for (int j = 0; j < kQd2dN; ++j) {
            const double y = phys_grid_.y.coord(j);
            for (int i = 0; i < kQd2dN; ++i) {
                const double x = phys_grid_.x.coord(i);
                v[static_cast<std::size_t>(phys_grid_.flat(i, j, 0))] =
                    0.5 * w0_ * w0_ * (x * x + y * y);
            }
        }
        prop_ = std::make_unique<ses::PeierlsLattice2D>(phys_grid_, v,
                                                        kQd2dDt);
        prop_->set_uniform_field(b_);
    }

    void push_trail() {
        double mx = 0.0;
        double my = 0.0;
        double den = 0.0;
        for (int j = 0; j < kQd2dN; ++j) {
            for (int i = 0; i < kQd2dN; ++i) {
                const double w = std::norm(psi_(i, j, 0));
                mx += phys_grid_.x.coord(i) * w;
                my += phys_grid_.y.coord(j) * w;
                den += w;
            }
        }
        trail_.push_back(static_cast<float>(mx / den));
        trail_.push_back(static_cast<float>(my / den));
        trail_.push_back(0.0f);
        if (trail_.size() > 3 * kQd2dTrailCap) {
            trail_.erase(trail_.begin(), trail_.begin() + 3);
        }
    }

    std::unique_ptr<ses::PeierlsLattice2D> prop_;
    ses::Heightfield hf_;
    bool mesh_dirty_ = false;
    double disp_peak_ = 0.0;
    std::vector<float> trail_;
    double w0_ = kQd2dW0;
    double b_ = kQd2dB;
    double last_e_ = 1e30;
    int conv_streak_ = 0;
    bool relaxing_ = false;
};

}  // namespace ses_shell
