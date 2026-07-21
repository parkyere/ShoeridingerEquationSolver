module;
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <utility>
#include <vector>
export module ses.scenario.billiard2d_director;
export import ses.scenario.lattice2d_director;
import ses.heightfield;
import ses.observables;
import ses.propagator;
import ses.parallel;
import ses.vk.engine_blobs;


// Quantum billiard: circle (integrable) vs Bunimovich stadium (chaotic).
// Spectral split-operator, B = 0. CONTRACT: tests/billiard2d_test.cpp.


export namespace ses_shell {

// Signed distance to stadium (half_len = 0 -> circle); negative inside.
inline double stadium_sdf(double x, double y, double half_len, double r) {
    const double cx = std::clamp(x, -half_len, half_len);
    return std::hypot(x - cx, y) - r;
}

constexpr double kBl2dBox = 30.0;
constexpr int kBl2dN = 512;
constexpr int kBl2dNz = 4;
constexpr double kBl2dZHalf = 2.0;
constexpr double kBl2dDt = 0.01;     // wall phase V0 dt = 0.24 rad/step
constexpr double kBl2dR = 13.0;
constexpr double kBl2dHalfLen = 6.5;
constexpr double kBl2dWallV0 = 24.0; // E = k0^2/2 = 2 Ha: near-total wall
constexpr double kBl2dWallW = 2.0;
constexpr double kBl2dK0 = 2.0;
constexpr double kBl2dSigma = 1.5;
constexpr double kBl2dLaunchY = 6.5; // (0, R/2): tangential family
constexpr int kBl2dStepsPerTick = 16;
constexpr double kBl2dSurfH = 6.0;
constexpr int kBl2dMeshStride = 1;

class Billiard2DDirector final : public Lattice2DDirectorBase,
                                 public BilliardApi {
public:
    Billiard2DDirector()
        : Lattice2DDirectorBase(kBl2dBox, kBl2dN, kBl2dNz, kBl2dZHalf) {
        rebuild_potential();
        fire();
    }

    // B = 0 spectral scene: the GPU path is the 3D split-operator Engine at
    // nz = 1 (VkFFT 2D), NOT the base Peierls lattice engine (a B=0 Peierls
    // tight-binding band would misdefine the dispersion). Planar step/CPU
    // parity is certified by vkcheck check_engine_planar.
    void init_compute(ses_vk::DeviceContext& ctx, bool device_ok,
                      std::int64_t /*free_vram*/) override {
        compute_attempted_ = true;
        eng_ok_ = device_ok &&
                  eng_.initialize(ctx, phys_grid_, ses_vk::engine_blobs(kBl2dN),
                                  v_, kBl2dDt, psi_.data());
        eng_dirty_ = false;  // initialize() uploaded the current psi_
    }
    void release_gpu() override {
        eng_.destroy();
        eng_ok_ = false;
        eng_dirty_ = true;
    }

    BilliardApi* billiard() override { return this; }

    // ---- BilliardApi ----
    void fire() override {
        ses::parallel_for(kBl2dN, [&](int j) {
            const double y = phys_grid_.y.coord(j) - kBl2dLaunchY;
            for (int i = 0; i < kBl2dN; ++i) {
                const double x = phys_grid_.x.coord(i);
                psi_(i, j, 0) =
                    std::exp(-(x * x + y * y) /
                             (4.0 * kBl2dSigma * kBl2dSigma)) *
                    std::complex<double>{std::cos(kBl2dK0 * x),
                                         std::sin(kBl2dK0 * x)};
            }
        });
        ses::normalize(psi_);
        avg_.assign(static_cast<std::size_t>(phys_grid_.size()), 0.0);
        avg_steps_ = 0;
        disp_peak_ = 0.0;
        eng_dirty_ = true;  // psi_ edited on the CPU: re-upload before stepping
        track_peak();
        mark_fired();
    }
    void toggle_shape() override {
        stadium_ = !stadium_;
        rebuild_potential();
        fire();
        title_dirty_ = true;
    }
    bool stadium() const override { return stadium_; }
    void toggle_avg_view() override {
        avg_view_ = !avg_view_;
        disp_peak_ = 0.0;  // re-snap: live and average scales differ
        staging_dirty_ = true;
        title_dirty_ = true;
    }
    bool avg_view() const override { return avg_view_; }
    // center/interior time-avg density ratio. CONTRACT: billiard2d_test.
    double avg_center_fraction() const override {
        if (avg_steps_ == 0) {
            return -1.0;
        }
        double center = 0.0;
        int n_center = 0;
        double interior = 0.0;
        int n_interior = 0;
        const double hl = stadium_ ? kBl2dHalfLen : 0.0;
        for (int j = 0; j < kBl2dN; ++j) {
            const double y = phys_grid_.y.coord(j);
            for (int i = 0; i < kBl2dN; ++i) {
                const double x = phys_grid_.x.coord(i);
                const double a =
                    avg_[static_cast<std::size_t>(j) * kBl2dN +
                         static_cast<std::size_t>(i)];
                if (std::hypot(x, y) < 2.0) {
                    center += a;
                    ++n_center;
                }
                if (stadium_sdf(x, y, hl, kBl2dR) < -2.0) {
                    interior += a;
                    ++n_interior;
                }
            }
        }
        center /= n_center;
        interior /= n_interior;
        return interior > 0.0 ? center / interior : -1.0;
    }

    void reset_simulation() override { fire(); }

    bool handle_key(char key) override {
        if (key == '2') {
            fire();
            return true;
        }
        if (key == '5') {
            toggle_shape();
            return true;
        }
        if (key == 'A') {
            toggle_avg_view();
            return true;
        }
        return false;
    }

    double sim_dt() const override { return kBl2dDt; }
    int steps_per_tick() const override { return kBl2dStepsPerTick; }

    // ---- surface display (corral rule) ----
    bool cloud() const override { return false; }
    const ses::Mesh& mesh() const override { return hf_.mesh; }
    const std::vector<ses::Rgb>& colors() const override {
        return hf_.colors;
    }
    bool take_mesh_dirty() override {
        return std::exchange(mesh_dirty_, false);
    }
    double default_camera_azimuth() const override { return 0.35; }
    double default_camera_elevation() const override { return 0.95; }
    double default_camera_distance() const override { return 110.0; }

    int overlay_curve_count() const override { return 1; }
    OverlayCurve overlay_curve(int /*i*/) const override {
        return {outline_.data(), static_cast<int>(outline_.size() / 3),
                0.85f, 0.85f, 0.90f, 0.5f};
    }

    std::string title_text() override {
        return strf(
            "Quantum billiard (%s)  |  t = %.1f au  view %s  "
            "center/interior = %.2f  keys: 2 fire / 5 shape / A average",
            stadium_ ? "Bunimovich stadium, CHAOTIC"
                     : "circle, integrable",
            sim_time_, avg_view_ ? "TIME AVERAGE (scar lens)" : "live",
            avg_center_fraction());
    }

protected:
    void rebuild_display() override {
        if (avg_view_ && avg_steps_ > 0) {
            // real field -> uniform phasor hue
            ses::parallel_for(kBl2dN, [&](int j) {
                for (int i = 0; i < kBl2dN; ++i) {
                    scar_(i, j, 0) = std::sqrt(
                        avg_[static_cast<std::size_t>(j) * kBl2dN +
                             static_cast<std::size_t>(i)] /
                        avg_steps_);
                }
            });
            double cur = 0.0;
            for (int j = 0; j < kBl2dN; ++j) {
                for (int i = 0; i < kBl2dN; ++i) {
                    cur = std::max(cur, std::norm(scar_(i, j, 0)));
                }
            }
            disp_peak_ = disp_peak_ <= 0.0
                             ? cur
                             : std::max(cur, 0.98 * disp_peak_);
            hf_ = ses::heightfield_surface(scar_, kBl2dSurfH, disp_peak_,
                                           kBl2dMeshStride);
            mesh_dirty_ = true;
            return;
        }
        double cur = 0.0;
        for (int j = 0; j < kBl2dN; ++j) {
            for (int i = 0; i < kBl2dN; ++i) {
                cur = std::max(cur, std::norm(psi_(i, j, 0)));
            }
        }
        disp_peak_ = disp_peak_ <= 0.0 ? cur
                                       : std::max(cur, 0.98 * disp_peak_);
        hf_ = ses::heightfield_surface(psi_, kBl2dSurfH, disp_peak_,
                                       kBl2dMeshStride);
        mesh_dirty_ = true;
    }

    void do_steps(int n) override {
        if (eng_ok_) {
            if (eng_dirty_) {
                eng_.upload_state(psi_.data());
                eng_dirty_ = false;
            }
            int left = n;
            while (left > 0) {
                const int chunk = std::min(left, 4);  // sample the avg 4x/tick
                eng_.step(chunk);
                eng_.readback(rb_);
                store_readback();
                accumulate_occupancy();
                ++avg_steps_;  // one time-average sample per chunk
                left -= chunk;
            }
        } else {
            for (int s = 0; s < n; ++s) {
                prop_->step(psi_, 1);
                accumulate_occupancy();
                ++avg_steps_;  // one sample per step
            }
        }
        sim_time_ += n * kBl2dDt;
        track_peak();
    }

private:
    void store_readback() {
        std::vector<std::complex<double>>& d = psi_.data();
        for (std::size_t i = 0; i < d.size(); ++i) {
            d[i] = std::complex<double>{static_cast<double>(rb_[2 * i]),
                                        static_cast<double>(rb_[2 * i + 1])};
        }
    }
    void accumulate_occupancy() {
        ses::parallel_for(kBl2dN, [&](int j) {
            const std::size_t row = static_cast<std::size_t>(j) * kBl2dN;
            for (int i = 0; i < kBl2dN; ++i) {
                avg_[row + static_cast<std::size_t>(i)] +=
                    std::norm(psi_(i, j, 0));
            }
        });
    }

    void rebuild_potential() {
        const double hl = stadium_ ? kBl2dHalfLen : 0.0;
        std::vector<double> v(
            static_cast<std::size_t>(phys_grid_.size()), 0.0);
        for (int j = 0; j < kBl2dN; ++j) {
            const double y = phys_grid_.y.coord(j);
            for (int i = 0; i < kBl2dN; ++i) {
                const double d =
                    stadium_sdf(phys_grid_.x.coord(i), y, hl, kBl2dR);
                if (d > 0.0) {
                    const double t = d / kBl2dWallW;
                    v[static_cast<std::size_t>(phys_grid_.flat(i, j, 0))] =
                        kBl2dWallV0 * std::min(1.0, t * t);
                }
            }
        }
        v_ = std::move(v);
        prop_ = std::make_unique<ses::SplitOperator3D>(phys_grid_, v_,
                                                       kBl2dDt);
        rebuild_outline(hl);
    }

    void rebuild_outline(double hl) {
        const double pi = 3.14159265358979323846;
        outline_.clear();
        auto put = [&](double th, double cx) {
            outline_.push_back(
                static_cast<float>(cx + kBl2dR * std::cos(th)));
            outline_.push_back(static_cast<float>(kBl2dR * std::sin(th)));
            outline_.push_back(0.0f);
        };
        for (int t = 0; t <= 24; ++t) {  // right cap
            put(-0.5 * pi + pi * t / 24.0, hl);
        }
        for (int t = 0; t <= 24; ++t) {  // left cap
            put(0.5 * pi + pi * t / 24.0, -hl);
        }
        put(-0.5 * pi, hl);  // close bottom flat
    }

    std::vector<double> v_;
    std::unique_ptr<ses::SplitOperator3D> prop_;  // CPU fallback
    ses_vk::Engine eng_;                          // GPU: planar split-operator
    bool eng_ok_ = false;
    bool eng_dirty_ = true;
    std::vector<float> rb_;                        // readback scratch
    ses::Heightfield hf_;
    bool mesh_dirty_ = false;
    ses::Field3D scar_{phys_grid_};
    std::vector<double> avg_;
    long long avg_steps_ = 0;
    std::vector<float> outline_;
    double disp_peak_ = 0.0;
    bool stadium_ = false;
    bool avg_view_ = false;
};

}  // namespace ses_shell
