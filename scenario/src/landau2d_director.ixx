module;
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <volk.h>
export module ses.scenario.landau2d_director;
export import ses.scenario;
export import ses.field;
export import ses.grid;
export import ses.lattice2d;
import ses.parallel;
import ses.potential;
import ses.vk.lattice2d_engine;


// Landau/cyclotron: 2D electron, uniform B||z. Peierls lattice,
// A_x = +B y gauge (gauge-exact), CCW orbit about the origin.


export namespace ses_shell {

constexpr double kLd2dBox = 30.0;
constexpr int kLd2dN = 512;
constexpr int kLd2dNz = 4;
constexpr double kLd2dZHalf = 2.0;
// dt=0.0025 holds the 256^2 bond angle tx*dt/2 at 512^2 (Trotter knob);
// StepsPerTick x4 offsets the dt cut to hold the visual pace.
constexpr double kLd2dDt = 0.0025;
constexpr double kLd2dB = 0.4;
constexpr double kLd2dBMin = 0.15;
constexpr double kLd2dBMax = 1.2;
constexpr double kLd2dK0 = 1.5;
constexpr double kLd2dK0Min = 0.5;
constexpr double kLd2dK0Max = 2.5;
constexpr double kLd2dSigma = 2.0;
constexpr int kLd2dStepsPerTick = 32;
constexpr int kLd2dTrailCap = 900;

class Landau2DDirector final : public ScenarioDirector, public LandauApi {
public:
    Landau2DDirector()
        : phys_grid_{ses::Grid1D{-kLd2dBox, kLd2dBox, kLd2dN},
                     ses::Grid1D{-kLd2dBox, kLd2dBox, kLd2dN},
                     ses::Grid1D{-1.0, 1.0, 1}},
          disp_grid_{ses::Grid1D{-kLd2dBox, kLd2dBox, kLd2dN},
                     ses::Grid1D{-kLd2dBox, kLd2dBox, kLd2dN},
                     ses::Grid1D{-kLd2dZHalf, kLd2dZHalf, kLd2dNz}},
          psi_{phys_grid_} {
        rebuild_prop();
        fire();
    }

    LandauApi* landau() override { return this; }

    // ---- LandauApi ----
    void set_field(double b) override {
        b_ = std::clamp(b, kLd2dBMin, kLd2dBMax);
        prop_->set_uniform_field(b_);
        if (eng_ok_) {
            eng_.set_uniform_field(b_);
        }
        fire();
    }
    double field() const override { return b_; }
    void set_k0(double k0) override {
        k0_ = std::clamp(k0, kLd2dK0Min, kLd2dK0Max);
        fire();
    }
    double k0() const override { return k0_; }
    void refire() override { fire(); }
    double omega_c() const override { return b_; }
    double radius_pred() const override { return k0_ / b_; }
    double orbit_x() const override { return mean_[0]; }
    double orbit_y() const override { return mean_[1]; }
    double mean_n() const override { return energy_ / b_ - 0.5; }
    double antipode_dist() const override { return antipode_dist_; }
    double closure_dist() const override { return closure_dist_; }
    // One quantum up/down; a level jump invalidates the old orbit records.
    // CONTRACT: lattice2d LandauLadder test.
    bool ladder(bool up) override {
        ses::Field3D next = ses::landau_ladder(psi_, b_, up);
        if (ses::norm_sq(next) < 1e-6) {
            return false;  // a|lowest> = 0: refuse the down-jump
        }
        ses::normalize(next);
        // Energy ceiling 0.30/h^2 = lattice band top (1D ladder_cap rule),
        // not a rung check: a-dag adds ~2B on the coherent orbit (only B on
        // eigenstates). CONTRACT: lattice2d_test LadderRefusesPastTheLatticeBand.
        const double e_cur = prop_->energy(psi_);
        const double e_next = prop_->energy(next);
        if (up) {
            const double h = phys_grid_.x.spacing();
            if (e_next > 0.30 / (h * h)) {
                return false;
            }
        } else if (e_cur - e_next < 0.05 * b_) {
            // a removes no clean quantum: coherent state is ~eigen of a
            // (a|alpha> = alpha|alpha>).
            return false;
        }
        psi_ = std::move(next);
        eng_dirty_ = true;  // psi_ replaced on the CPU: re-upload before stepping
        measure();
        trail_.clear();
        push_trail();
        antipode_dist_ = -1.0;
        closure_dist_ = -1.0;
        display_changed_ = true;
        vol_dirty_ = true;
        staging_dirty_ = true;
        title_dirty_ = true;
        return true;
    }

    // ---- lifecycle ----
    const ses::Grid3D& grid() const override { return disp_grid_; }
    // Peierls scene (uniform B): the GPU port is ses_vk::Lattice2DEngine
    // (vkcheck-certified). The CPU prop_ stays for the link-resolved energy in
    // measure() and as the fallback; only the propagation moves to the GPU.
    void init_compute(ses_vk::DeviceContext& ctx, bool device_ok,
                      std::int64_t /*free_vram*/) override {
        compute_attempted_ = true;
        if (device_ok && eng_.initialize(ctx)) {
            eng_.set_lattice(
                phys_grid_,
                std::vector<double>(
                    static_cast<std::size_t>(phys_grid_.size()), 0.0),
                kLd2dDt);
            eng_.set_uniform_field(b_);
            eng_.set_absorber(mask_);
            eng_.upload(psi_.data());
            eng_ok_ = true;
            eng_dirty_ = false;
        }
    }
    void release_gpu() override {
        eng_.destroy();
        eng_ok_ = false;
        eng_dirty_ = true;
    }
    bool compute_attempted() const override { return compute_attempted_; }
    bool gpu_ok() const override { return false; }

    // ---- per-frame ----
    void run_frame() override {
        if (pending_steps_ > 0) {
            int n = pending_steps_;
            pending_steps_ = 0;
            // Chunk so trail + antipode/closure records get finer
            // granularity than one time-scaled batch.
            if (eng_ok_ && eng_dirty_) {
                eng_.upload(psi_.data());
                eng_dirty_ = false;
            }
            while (n > 0) {
                const int chunk = std::min(n, 8);
                if (eng_ok_) {
                    eng_.step(chunk);  // GPU applies the edge absorber per step
                    eng_.download();
                    store_engine_state();
                    // contained orbit never reaches the frame absorber, so the
                    // GPU norm stays ~1; normalize the display/measure copy only.
                    ses::normalize(psi_);
                } else {
                    prop_->step(psi_, chunk);
                    ses::parallel_for(static_cast<int>(mask_.size()),
                                      [&](int c) {
                                          psi_.data()[static_cast<std::size_t>(
                                              c)] *=
                                              mask_[static_cast<std::size_t>(c)];
                                      });
                    ses::normalize(psi_);
                }
                sim_time_ += chunk * kLd2dDt;
                n -= chunk;
                measure();
                push_trail();
                record_crossings();
            }
            display_changed_ = true;
            vol_dirty_ = true;
            staging_dirty_ = true;
        }
        if (staging_dirty_) {
            staging_dirty_ = false;
            rebuild_staging();
        }
        if (++frames_ % 10 == 0) {
            title_dirty_ = true;
        }
    }
    void tick() override {
        const int per_tick = kLd2dStepsPerTick * time_scale_;
        pending_steps_ = std::min(pending_steps_ + per_tick, per_tick);
    }

    // ---- controls ----
    void do_set_real_time() override {}
    void reset_simulation() override { fire(); }
    void measure_now() override {}
    void toggle_view_mode() override {}
    bool handle_key(char key) override {
        if (key == '2') {
            fire();
            return true;
        }
        if (key == '3') {
            return ladder(true);
        }
        if (key == '4') {
            return ladder(false);
        }
        return false;
    }

    bool solving() const override { return false; }
    bool scene_ready() const override { return true; }
    void set_time_scale(int scale) override {
        time_scale_ = std::clamp(scale, 1, 16);
    }
    int time_scale() const override { return time_scale_; }
    double sim_time() const override { return sim_time_; }
    double sim_dt() const override { return kLd2dDt; }
    int steps_per_tick_x1() const override { return kLd2dStepsPerTick; }

    // ---- display ----
    bool cloud() const override { return true; }
    double peak() const override { return peak_; }
    VkImageView psi_volume_view() override { return VK_NULL_HANDLE; }
    float next_flash_intensity() override { return 0.0f; }
    bool take_volume_written() override {
        return std::exchange(display_changed_, false);
    }
    bool take_volume_dirty() override {
        return std::exchange(vol_dirty_, false);
    }
    bool take_mesh_dirty() override { return false; }
    void mark_display_dirty() override {
        display_changed_ = true;
        vol_dirty_ = true;
    }
    bool take_title_dirty() override {
        return std::exchange(title_dirty_, false);
    }
    const std::vector<float>& psi_staging() const override {
        return staging_;
    }
    const ses::Mesh& mesh() const override { return no_mesh_; }
    const std::vector<ses::Rgb>& colors() const override {
        return no_colors_;
    }
    std::string title_text() override {
        const double pi = 3.14159265358979323846;
        return strf(
            "Landau levels / cyclotron (2D lattice, uniform B)  |  t = %.1f "
            "au  B = %.2f  k0 = %.2f  r = %.2f (pred %.2f)  T = %.1f  "
            "<n> = %.1f  keys: 2 refire / 3 up / 4 down",
            sim_time_, b_, k0_,
            std::hypot(mean_[0] - center_[0], mean_[1] - center_[1]),
            radius_pred(), 2.0 * pi / b_, mean_n());
    }

    int marker_count() const override { return 0; }

    int overlay_curve_count() const override { return 2; }
    OverlayCurve overlay_curve(int i) const override {
        if (i == 0) {
            return {orbit_curve_.data(),
                    static_cast<int>(orbit_curve_.size() / 3),
                    1.0f, 0.70f, 0.20f, 0.9f};
        }
        return {trail_.data(), static_cast<int>(trail_.size() / 3),
                1.0f, 1.0f, 1.0f, 0.9f};
    }

    double default_camera_azimuth() const override { return 0.0; }
    double default_camera_elevation() const override { return 0.0; }
    double default_camera_distance() const override { return 95.0; }

private:
    void rebuild_prop() {
        const std::vector<double> zero(
            static_cast<std::size_t>(phys_grid_.size()), 0.0);
        prop_ = std::make_unique<ses::PeierlsLattice2D>(phys_grid_, zero,
                                                        kLd2dDt);
        prop_->set_uniform_field(b_);
        // Open plane, not a torus: absorb tunneled amplitude at the frame,
        // renormalize each chunk (corral rule).
        mask_ = ses::absorbing_mask(phys_grid_, 4.0);
    }

    // Launch on the y=0 row (mechanical = canonical momentum in this gauge),
    // tangentially -> orbit circles the origin.
    void fire() {
        const double x0 = radius_pred();
        ses::parallel_for(kLd2dN, [&](int j) {
            const double y = phys_grid_.y.coord(j);
            for (int i = 0; i < kLd2dN; ++i) {
                const double x = phys_grid_.x.coord(i);
                const double dx = x - x0;
                const double env =
                    std::exp(-(dx * dx + y * y) /
                             (4.0 * kLd2dSigma * kLd2dSigma));
                psi_(i, j, 0) =
                    env * std::complex<double>{std::cos(k0_ * y),
                                               std::sin(k0_ * y)};
            }
        });
        ses::normalize(psi_);
        center_[0] = 0.0;
        center_[1] = 0.0;
        sim_time_ = 0.0;
        pending_steps_ = 0;
        antipode_dist_ = -1.0;
        closure_dist_ = -1.0;
        trail_.clear();
        measure();
        push_trail();
        rebuild_orbit_overlay();
        double pk = 0.0;
        for (int j = 0; j < kLd2dN; ++j) {
            for (int i = 0; i < kLd2dN; ++i) {
                pk = std::max(pk, std::norm(psi_(i, j, 0)));
            }
        }
        peak_ = pk;
        eng_dirty_ = true;  // psi_ rebuilt on the CPU: re-upload before stepping
        display_changed_ = true;
        vol_dirty_ = true;
        staging_dirty_ = true;
        title_dirty_ = true;
    }

    void measure() {
        double mx = 0.0;
        double my = 0.0;
        double den = 0.0;
        double e_hop = 0.0;
        double e_site = 0.0;
        const double hx = phys_grid_.x.spacing();
        const double hy = phys_grid_.y.spacing();
        const double tx = 0.5 / (hx * hx);
        const double ty = 0.5 / (hy * hy);
        for (int j = 0; j < kLd2dN; ++j) {
            for (int i = 0; i < kLd2dN; ++i) {
                const std::complex<double> z = psi_(i, j, 0);
                const double w = std::norm(z);
                mx += phys_grid_.x.coord(i) * w;
                my += phys_grid_.y.coord(j) * w;
                den += w;
                e_site += 2.0 * (tx + ty) * w;
                if (i + 1 < kLd2dN) {
                    e_hop += -tx * 2.0 *
                             (std::conj(z) * prop_->link_x(i, j) *
                              psi_(i + 1, j, 0))
                                 .real();
                }
                if (j + 1 < kLd2dN) {
                    e_hop += -ty * 2.0 *
                             (std::conj(z) * prop_->link_y(i, j) *
                              psi_(i, j + 1, 0))
                                 .real();
                }
            }
        }
        mean_[0] = mx / den;
        mean_[1] = my / den;
        energy_ = (e_hop + e_site) / den;
    }

    void record_crossings() {
        const double pi = 3.14159265358979323846;
        const double period = 2.0 * pi / b_;
        const double r = radius_pred();
        if (antipode_dist_ < 0.0 && sim_time_ >= 0.5 * period) {
            antipode_dist_ = std::hypot(mean_[0] + r, mean_[1]);
        }
        if (closure_dist_ < 0.0 && sim_time_ >= period) {
            closure_dist_ = std::hypot(mean_[0] - r, mean_[1]);
        }
    }

    void push_trail() {
        trail_.push_back(static_cast<float>(mean_[0]));
        trail_.push_back(static_cast<float>(mean_[1]));
        trail_.push_back(0.0f);
        if (trail_.size() > 3 * kLd2dTrailCap) {
            trail_.erase(trail_.begin(), trail_.begin() + 3);
        }
    }

    void rebuild_orbit_overlay() {
        const double pi = 3.14159265358979323846;
        orbit_curve_.clear();
        auto put = [&](double x, double y) {
            orbit_curve_.push_back(static_cast<float>(x));
            orbit_curve_.push_back(static_cast<float>(y));
            orbit_curve_.push_back(0.0f);
        };
        const double r = radius_pred();
        for (int t = 0; t <= 64; ++t) {
            const double th = 2.0 * pi * t / 64.0;
            put(center_[0] + r * std::cos(th),
                center_[1] + r * std::sin(th));
        }
        // Center cross, retraced from the circle's seam.
        put(center_[0] - 0.8, center_[1]);
        put(center_[0] + 0.8, center_[1]);
        put(center_[0], center_[1]);
        put(center_[0], center_[1] - 0.8);
        put(center_[0], center_[1] + 0.8);
    }

    void rebuild_staging() {
        const std::size_t plane = static_cast<std::size_t>(kLd2dN) *
                                  static_cast<std::size_t>(kLd2dN);
        staging_.resize(plane * kLd2dNz * 2);
        ses::parallel_for(kLd2dN, [&](int j) {
            for (int i = 0; i < kLd2dN; ++i) {
                const std::complex<double> z = psi_(i, j, 0);
                const std::size_t o =
                    2 * (static_cast<std::size_t>(j) *
                             static_cast<std::size_t>(kLd2dN) +
                         static_cast<std::size_t>(i));
                staging_[o] = static_cast<float>(z.real());
                staging_[o + 1] = static_cast<float>(z.imag());
            }
        });
        for (int k = 1; k < kLd2dNz; ++k) {
            std::copy(staging_.begin(),
                      staging_.begin() +
                          static_cast<std::ptrdiff_t>(plane * 2),
                      staging_.begin() +
                          static_cast<std::ptrdiff_t>(plane * 2) * k);
        }
    }

    void store_engine_state() {
        const float* s = eng_.state();
        std::vector<std::complex<double>>& d = psi_.data();
        for (std::size_t i = 0; i < d.size(); ++i) {
            d[i] = std::complex<double>{static_cast<double>(s[2 * i]),
                                        static_cast<double>(s[2 * i + 1])};
        }
    }

    static std::string strf(const char* fmt, ...) {
        char buf[512];
        va_list args;
        va_start(args, fmt);
        std::vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        return std::string{buf};
    }

    ses::Grid3D phys_grid_;
    ses::Grid3D disp_grid_;
    ses::Field3D psi_;
    std::vector<double> mask_;  // open-plane absorber frame
    std::unique_ptr<ses::PeierlsLattice2D> prop_;  // CPU: links/energy + fallback
    ses_vk::Lattice2DEngine eng_;                  // GPU Peierls propagator
    bool eng_ok_ = false;
    bool eng_dirty_ = true;
    std::vector<float> staging_;
    std::vector<float> orbit_curve_;
    std::vector<float> trail_;
    double b_ = kLd2dB;
    double k0_ = kLd2dK0;
    double mean_[2] = {0.0, 0.0};
    double center_[2] = {0.0, 0.0};
    double antipode_dist_ = -1.0;
    double closure_dist_ = -1.0;
    double energy_ = 0.0;
    double peak_ = 1.0;
    double sim_time_ = 0.0;
    int pending_steps_ = 0;
    int time_scale_ = 1;
    std::uint64_t frames_ = 0;
    bool display_changed_ = true;
    bool vol_dirty_ = true;
    bool staging_dirty_ = true;
    bool title_dirty_ = true;
    bool compute_attempted_ = false;

    ses::Mesh no_mesh_;
    std::vector<ses::Rgb> no_colors_;
};

}  // namespace ses_shell
