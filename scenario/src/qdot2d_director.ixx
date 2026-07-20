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
// Translucent red PARABOLOID: z = kQd2dEScale V(r), capped at ParaTop.
constexpr double kQd2dEScale = 0.5;
constexpr double kQd2dParaTop = 11.0;
constexpr int kQd2dParaRings = 14;
constexpr int kQd2dParaSegs = 48;

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

    // Circular ladder (keys 3/4): a_R-dag adds one w0 quantum (+1 L_z).
    // At B != 0 the lattice gauge is NOT the symmetric gauge the circular
    // ladder lives in -- honest refuse, not silent wrongness.
    // CONTRACT: tests/ho2d_test.cpp (core op).
    bool ho_ladder(bool up) override {
        if (relaxing_) {
            return false;
        }
        if (b_ != 0.0) {
            note_ = "ladder needs B = 0 (gauge)";
            title_dirty_ = true;
            return false;
        }
        ses::Field3D next = ses::ho2d_ladder(psi_, w0_, up);
        if (ses::norm_sq(next) < 1e-6) {
            note_ = "a|0> = 0: refused";
            title_dirty_ = true;
            return false;
        }
        ses::normalize(next);
        psi_ = std::move(next);
        note_.clear();
        trail_.clear();
        track_peak();
        staging_dirty_ = true;
        display_changed_ = true;
        title_dirty_ = true;
        return true;
    }

    // Random coherent packet (key S): displaced Gaussian with a random
    // kick -- a fresh classical orbit each press.
    void random_packet() override {
        relaxing_ = false;
        std::uniform_real_distribution<double> ang(0.0,
                                                   6.28318530717958647692);
        std::uniform_real_distribution<double> rad(2.0, 5.0);
        std::uniform_real_distribution<double> kick(0.0, 2.0 * w0_);
        const double th = ang(rng_);
        const double r0 = rad(rng_);
        const double x0 = r0 * std::cos(th);
        const double y0 = r0 * std::sin(th);
        const double kth = ang(rng_);
        const double kk = kick(rng_);
        const double kx = kk * std::cos(kth);
        const double ky = kk * std::sin(kth);
        const double om = energy_pred();  // Omega sets the coherent width
        ses::parallel_for(kQd2dN, [&](int j) {
            const double y = phys_grid_.y.coord(j) - y0;
            for (int i = 0; i < kQd2dN; ++i) {
                const double x = phys_grid_.x.coord(i) - x0;
                const double ph = kx * phys_grid_.x.coord(i) +
                                  ky * phys_grid_.y.coord(j);
                psi_(i, j, 0) =
                    std::exp(-0.5 * om * (x * x + y * y)) *
                    std::complex<double>{std::cos(ph), std::sin(ph)};
            }
        });
        ses::normalize(psi_);
        note_.clear();
        trail_.clear();
        disp_peak_ = 0.0;
        track_peak();
        mark_fired();
    }

    // ---- interactive grab (the pick-and-gather toy) ----
    // While grabbed the evolution PAUSES; the display blends the held
    // state with a coherent Gaussian at the grab point -- pull higher,
    // gather harder. Release: the blend IS the new state, time resumes.
    void begin_grab(double x, double y) override {
        relaxing_ = false;
        grab_x_ = std::clamp(x, -kQd2dBox, kQd2dBox);
        grab_y_ = std::clamp(y, -kQd2dBox, kQd2dBox);
        grab_base_ = psi_;
        grabbing_ = true;
        note_ = "grabbed";
        title_dirty_ = true;
    }
    void update_grab(double strength) override {
        if (!grabbing_) {
            return;
        }
        const double s = std::clamp(strength, 0.0, 1.0);
        const double om = energy_pred();
        ses::Field3D gath{phys_grid_};
        ses::parallel_for(kQd2dN, [&](int j) {
            const double y = phys_grid_.y.coord(j) - grab_y_;
            for (int i = 0; i < kQd2dN; ++i) {
                const double x = phys_grid_.x.coord(i) - grab_x_;
                gath(i, j, 0) = std::exp(-0.5 * om * (x * x + y * y));
            }
        });
        ses::normalize(gath);
        for (std::size_t c = 0; c < psi_.data().size(); ++c) {
            psi_.data()[c] =
                (1.0 - s) * grab_base_.data()[c] + s * gath.data()[c];
        }
        ses::normalize(psi_);
        track_peak();
        staging_dirty_ = true;
        display_changed_ = true;
    }
    void end_grab() override {
        if (!grabbing_) {
            return;
        }
        grabbing_ = false;
        note_.clear();
        trail_.clear();
        title_dirty_ = true;
    }
    bool grabbing() const override { return grabbing_; }

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
        if (key == '3') {
            ho_ladder(true);
            return true;
        }
        if (key == '4') {
            ho_ladder(false);
            return true;
        }
        if (key == 'S') {
            random_packet();
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
        if (!note_.empty()) {
            s += "  [" + note_ + "]";
        }
        s += "  keys: 2 ground / F displace / 3 up / 4 down / S random / "
             "drag the surface to gather";
        return s;
    }

    // 0 = the translucent red potential paraboloid (drawn first),
    // 1 = the measured <r>(t) breadcrumb trail (the rosette).
    int overlay_curve_count() const override { return 2; }
    OverlayCurve overlay_curve(int i) const override {
        if (i == 0) {
            return {para_.data(), static_cast<int>(para_.size() / 3),
                    1.0f, 0.30f, 0.25f, 0.28f, true};
        }
        return {trail_.data(), static_cast<int>(trail_.size() / 3),
                1.0f, 1.0f, 1.0f, 0.9f};
    }

protected:
    void do_steps(int n) override {
        if (grabbing_) {
            return;  // held: time stands still until release
        }
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
        rebuild_paraboloid();
    }

    // Concentric triangle-strip rings up to the display cap; z rides the
    // SAME e-scale the well would, so a stiffer w0 visibly narrows it.
    void rebuild_paraboloid() {
        para_.clear();
        const double pi = 3.14159265358979323846;
        const double r_top = std::min(
            0.95 * kQd2dBox,
            std::sqrt(2.0 * kQd2dParaTop / (kQd2dEScale * w0_ * w0_)));
        auto put = [&](double r, double th) {
            para_.push_back(static_cast<float>(r * std::cos(th)));
            para_.push_back(static_cast<float>(r * std::sin(th)));
            para_.push_back(static_cast<float>(
                kQd2dEScale * 0.5 * w0_ * w0_ * r * r));
        };
        for (int k = 0; k < kQd2dParaRings; ++k) {
            const double r0 = r_top * k / kQd2dParaRings;
            const double r1 = r_top * (k + 1) / kQd2dParaRings;
            if (!para_.empty()) {
                // degenerate bridge: repeat last vertex + the next head
                const std::size_t n = para_.size();
                para_.push_back(para_[n - 3]);
                para_.push_back(para_[n - 2]);
                para_.push_back(para_[n - 1]);
                put(r0, 0.0);
            }
            for (int t = 0; t <= kQd2dParaSegs; ++t) {
                const double th = 2.0 * pi * t / kQd2dParaSegs;
                put(r0, th);
                put(r1, th);
            }
        }
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
    std::string note_;
    std::mt19937 rng_{20260720u};
    std::vector<float> para_;
    // Grab state: held base + pick point (the pick-and-gather toy).
    bool grabbing_ = false;
    double grab_x_ = 0.0;
    double grab_y_ = 0.0;
    ses::Field3D grab_base_{phys_grid_};
};

}  // namespace ses_shell
