module;
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <volk.h>
export module ses.scenario.spins_director;
export import ses.scenario;
export import ses.spinlattice;


// 25 pinned electron spins, INTERACTING through the mean-field
// Heisenberg exchange (ses.spinlattice: product ansatz -- honestly a
// quantum-dressed classical Heisenberg/LLG lattice, no entanglement):
// each site is a small Bloch sphere with its own <sigma> arrow. J > 0
// orders ferro (watch a random boot comb itself into alignment under
// damping), J < 0 weaves the Neel checkerboard; alpha = 0 shows spin
// waves rippling forever; [M] Born-projects EVERY site onto +-B_hat.
// CONTRACT: tests/spinlattice_test.cpp (core) + --selftest-spins.
// volk.h textually first: VK_* macros never cross module boundaries.


export namespace ses_shell {

constexpr int kSlN = 5;              // 5 x 5
constexpr double kSlDt = 0.002;
constexpr int kSlStepsPerTick = 20;
constexpr double kSlR = 3.2;         // per-site sphere radius
constexpr double kSlPitch = 8.0;     // lattice spacing on screen
constexpr double kSlJ = 0.5;
constexpr double kSlAlpha = 0.05;

class SpinsDirector final : public ScenarioDirector, public SpinsApi {
public:
    SpinsDirector() {
        lat_.nx = kSlN;
        lat_.ny = kSlN;
        lat_.s.resize(kSlN * kSlN);
        rebuild_rings();
        seed_random();
    }

    SpinsApi* spins() override { return this; }

    // ---- SpinsApi ----
    void set_j(double j) override {
        j_ = std::clamp(j, -1.0, 1.0);
        title_dirty_ = true;
    }
    double j() const override { return j_; }
    void set_alpha(double a) override {
        alpha_ = std::clamp(a, 0.0, 0.3);
        title_dirty_ = true;
    }
    double alpha() const override { return alpha_; }
    void set_b(int axis, double v) override {
        b_[axis] = std::clamp(v, -1.0, 1.0);
        title_dirty_ = true;
    }
    double b(int axis) const override { return b_[axis]; }
    void seed_random() override {
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        std::uniform_real_distribution<double> ph(
            0.0, 6.28318530717958647692);
        for (auto& s : lat_.s) {
            const double z = u(rng_);
            const double t = ph(rng_);
            const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
            s = ses::spinor_from_bloch(r * std::cos(t), r * std::sin(t),
                                       z);
        }
        after_seed("random");
    }
    void seed_ferro() override {
        for (auto& s : lat_.s) {
            s = ses::spinor_from_bloch(1.0, 0.0, 0.0);
        }
        after_seed("ferro +x");
    }
    void seed_neel() override {
        for (int y = 0; y < kSlN; ++y) {
            for (int x = 0; x < kSlN; ++x) {
                const double sgn = ((x + y) & 1) != 0 ? -1.0 : 1.0;
                lat_.s[static_cast<std::size_t>(y * kSlN + x)] =
                    ses::spinor_from_bloch(0.0, 0.0, sgn);
            }
        }
        after_seed("Neel");
    }
    double magnetization() override {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        ses::lattice_magnetization(lat_, &x, &y, &z);
        return std::sqrt(x * x + y * y + z * z);
    }
    double staggered() override { return ses::lattice_staggered(lat_); }

    // ---- lifecycle / frame ----
    const ses::Grid3D& grid() const override { return grid_; }
    void init_compute(ses_vk::DeviceContext&, bool, std::int64_t) override {
        compute_attempted_ = true;
    }
    void release_gpu() override {}
    bool compute_attempted() const override { return compute_attempted_; }
    bool gpu_ok() const override { return false; }

    void run_frame() override {
        if (pending_steps_ > 0) {
            const int n = pending_steps_;
            pending_steps_ = 0;
            for (int k = 0; k < n; ++k) {
                ses::spinlattice_step(lat_, b_[0], b_[1], b_[2], j_,
                                      alpha_, kSlDt);
            }
            sim_time_ += n * kSlDt;
            display_changed_ = true;
        }
        rebuild_arrows();
        if (++frames_ % 10 == 0) {
            title_dirty_ = true;
        }
    }
    void tick() override {
        const int per_tick = kSlStepsPerTick * time_scale_;
        pending_steps_ = std::min(pending_steps_ + per_tick, per_tick);
    }

    void do_set_real_time() override {}
    void reset_simulation() override { seed_random(); }
    // [M]: Born-project every site onto +-B_hat (or +-z at B = 0).
    void measure_now() override {
        double nx = b_[0];
        double ny = b_[1];
        double nz = b_[2];
        const double bn = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (bn < 1e-9) {
            nx = 0.0;
            ny = 0.0;
            nz = 1.0;
        } else {
            nx /= bn;
            ny /= bn;
            nz /= bn;
        }
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        int plus = 0;
        for (auto& s : lat_.s) {
            plus += ses::spin_measure(s, nx, ny, nz, uni(rng_)) > 0 ? 1
                                                                    : 0;
        }
        note_ = strf("measured: %d/%d aligned", plus, kSlN * kSlN);
        display_changed_ = true;
        title_dirty_ = true;
    }
    void toggle_view_mode() override {}
    bool handle_key(char key) override {
        if (key == '2') {
            seed_random();
            return true;
        }
        if (key == '3') {
            seed_ferro();
            return true;
        }
        if (key == '4') {
            seed_neel();
            return true;
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
    double sim_dt() const override { return kSlDt; }
    int steps_per_tick_x1() const override { return kSlStepsPerTick; }

    // ---- display (overlay-only) ----
    bool cloud() const override { return false; }
    double peak() const override { return 1.0; }
    VkImageView psi_volume_view() override { return VK_NULL_HANDLE; }
    float next_flash_intensity() override { return 0.0f; }
    bool take_volume_written() override {
        return std::exchange(display_changed_, false);
    }
    bool take_volume_dirty() override { return false; }
    bool take_mesh_dirty() override { return false; }
    void mark_display_dirty() override { display_changed_ = true; }
    bool take_title_dirty() override {
        return std::exchange(title_dirty_, false);
    }
    const std::vector<float>& psi_staging() const override {
        return no_staging_;
    }
    const ses::Mesh& mesh() const override { return no_mesh_; }
    const std::vector<ses::Rgb>& colors() const override {
        return no_colors_;
    }
    int marker_count() const override { return 0; }

    std::string title_text() override {
        std::string s = strf(
            "25 interacting spins (mean-field Heisenberg -- product "
            "ansatz, no entanglement)  |  t = %.1f au  J = %+.2f  "
            "alpha = %.2f  |M| = %.2f  Neel = %.2f",
            sim_time_, j_, alpha_, magnetization(), staggered());
        if (!note_.empty()) {
            s += "  [" + note_ + "]";
        }
        s += "  keys: 2 random / 3 ferro / 4 Neel / M measure";
        return s;
    }

    // 2 rings per site (static) + 1 arrow per site (live).
    int overlay_curve_count() const override {
        return 3 * kSlN * kSlN;
    }
    OverlayCurve overlay_curve(int i) const override {
        const int site = i / 3;
        const int part = i % 3;
        if (part < 2) {
            return {&rings_[static_cast<std::size_t>(2 * site + part)][0],
                    static_cast<int>(
                        rings_[static_cast<std::size_t>(2 * site + part)]
                            .size() /
                        3),
                    0.75f, 0.80f, 0.95f, 0.22f};
        }
        return {&arrows_[static_cast<std::size_t>(site)][0],
                static_cast<int>(
                    arrows_[static_cast<std::size_t>(site)].size() / 3),
                1.0f, 0.75f, 0.20f, 1.0f};
    }

    double default_camera_azimuth() const override { return 0.35; }
    double default_camera_elevation() const override { return 0.75; }
    double default_camera_distance() const override { return 80.0; }

private:
    static constexpr double kPi = 3.14159265358979323846;

    static void site_center(int site, double* cx, double* cy) {
        const int x = site % kSlN;
        const int y = site / kSlN;
        *cx = (x - (kSlN - 1) / 2.0) * kSlPitch;
        *cy = (y - (kSlN - 1) / 2.0) * kSlPitch;
    }

    void after_seed(const char* what) {
        sim_time_ = 0.0;
        pending_steps_ = 0;
        note_ = what;
        display_changed_ = true;
        title_dirty_ = true;
    }

    void rebuild_rings() {
        rings_.assign(2 * kSlN * kSlN, {});
        for (int site = 0; site < kSlN * kSlN; ++site) {
            double cx = 0.0;
            double cy = 0.0;
            site_center(site, &cx, &cy);
            for (int part = 0; part < 2; ++part) {
                std::vector<float>& r =
                    rings_[static_cast<std::size_t>(2 * site + part)];
                for (int t = 0; t <= 32; ++t) {
                    const double th = 2.0 * kPi * t / 32.0;
                    const double a = kSlR * std::cos(th);
                    const double b = kSlR * std::sin(th);
                    if (part == 0) {  // ring about z (xy circle)
                        r.push_back(static_cast<float>(cx + a));
                        r.push_back(static_cast<float>(cy + b));
                        r.push_back(0.0f);
                    } else {  // ring about x (yz circle)
                        r.push_back(static_cast<float>(cx));
                        r.push_back(static_cast<float>(cy + a));
                        r.push_back(static_cast<float>(b));
                    }
                }
            }
        }
    }

    void rebuild_arrows() {
        arrows_.assign(static_cast<std::size_t>(kSlN * kSlN), {});
        for (int site = 0; site < kSlN * kSlN; ++site) {
            double cx = 0.0;
            double cy = 0.0;
            site_center(site, &cx, &cy);
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            ses::bloch_vector(lat_.s[static_cast<std::size_t>(site)], &x,
                              &y, &z);
            std::vector<float>& a =
                arrows_[static_cast<std::size_t>(site)];
            a.push_back(static_cast<float>(cx));
            a.push_back(static_cast<float>(cy));
            a.push_back(0.0f);
            a.push_back(static_cast<float>(cx + kSlR * x));
            a.push_back(static_cast<float>(cy + kSlR * y));
            a.push_back(static_cast<float>(kSlR * z));
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

    ses::Grid3D grid_{ses::Grid1D{-25.0, 25.0, 2},
                      ses::Grid1D{-25.0, 25.0, 2},
                      ses::Grid1D{-25.0, 25.0, 2}};
    ses::SpinLattice lat_;
    double b_[3] = {0.0, 0.0, 0.2};
    double j_ = kSlJ;
    double alpha_ = kSlAlpha;
    double sim_time_ = 0.0;
    int pending_steps_ = 0;
    int time_scale_ = 1;
    std::uint64_t frames_ = 0;
    bool display_changed_ = true;
    bool title_dirty_ = true;
    bool compute_attempted_ = false;
    std::string note_;
    std::mt19937 rng_{20260722u};
    std::vector<std::vector<float>> rings_;
    std::vector<std::vector<float>> arrows_;
    ses::Mesh no_mesh_;
    std::vector<ses::Rgb> no_colors_;
    std::vector<float> no_staging_;
};

}  // namespace ses_shell
