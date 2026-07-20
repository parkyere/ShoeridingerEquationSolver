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
export import ses.spinexact;


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

constexpr int kSlN = 4;              // 4 x 4 (= the exact 2^16 stage)
constexpr double kSlDt = 0.002;
constexpr int kSlStepsPerTick = 20;
constexpr int kSlExactSteps = 8;     // exact steps cost ~1.7 ms apiece
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
        for (int i = 0; i < kSlN * kSlN; ++i) {
            x += bloch_[static_cast<std::size_t>(3 * i)];
            y += bloch_[static_cast<std::size_t>(3 * i + 1)];
            z += bloch_[static_cast<std::size_t>(3 * i + 2)];
        }
        const double inv = 1.0 / (kSlN * kSlN);
        return std::sqrt(x * x + y * y + z * z) * inv;
    }
    double staggered() override {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        for (int yy = 0; yy < kSlN; ++yy) {
            for (int xx = 0; xx < kSlN; ++xx) {
                const int i = yy * kSlN + xx;
                const double sgn = ((xx + yy) & 1) != 0 ? -1.0 : 1.0;
                x += sgn * bloch_[static_cast<std::size_t>(3 * i)];
                y += sgn * bloch_[static_cast<std::size_t>(3 * i + 1)];
                z += sgn * bloch_[static_cast<std::size_t>(3 * i + 2)];
            }
        }
        const double inv = 1.0 / (kSlN * kSlN);
        return std::sqrt(x * x + y * y + z * z) * inv;
    }

    // ---- the exact/mean-field switch ----
    // Exact -> mean-field: each arrow COLLAPSES to a unit product spin
    // (direction kept, entanglement discarded -- that is the honest
    // meaning of the ansatz). Mean-field -> exact: the product embeds
    // losslessly.
    void set_exact(bool on) override {
        if (on == exact_mode_) {
            return;
        }
        if (on) {
            exact_ = ses::exact_from_product(lat_);
        } else {
            refresh_bloch();
            for (int i = 0; i < kSlN * kSlN; ++i) {
                double x = bloch_[static_cast<std::size_t>(3 * i)];
                double y = bloch_[static_cast<std::size_t>(3 * i + 1)];
                double z = bloch_[static_cast<std::size_t>(3 * i + 2)];
                const double n = std::sqrt(x * x + y * y + z * z);
                if (n > 1e-9) {
                    x /= n;
                    y /= n;
                    z /= n;
                } else {
                    x = 0.0;
                    y = 0.0;
                    z = 1.0;
                }
                lat_.s[static_cast<std::size_t>(i)] =
                    ses::spinor_from_bloch(x, y, z);
            }
        }
        exact_mode_ = on;
        note_ = on ? "EXACT 2^16 Hamiltonian" : "mean-field (product)";
        title_dirty_ = true;
    }
    bool exact_mode() const override { return exact_mode_; }
    double arrow_mean() override {
        double m = 0.0;
        for (int i = 0; i < kSlN * kSlN; ++i) {
            const double x = bloch_[static_cast<std::size_t>(3 * i)];
            const double y = bloch_[static_cast<std::size_t>(3 * i + 1)];
            const double z = bloch_[static_cast<std::size_t>(3 * i + 2)];
            m += std::sqrt(x * x + y * y + z * z);
        }
        return m / (kSlN * kSlN);
    }

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
                if (exact_mode_) {
                    // Exact mode is a CLOSED quantum system: alpha does
                    // not apply (dissipation needs an environment).
                    ses::exact_step(exact_, b_[0], b_[1], b_[2], j_,
                                    kSlDt);
                } else {
                    ses::spinlattice_step(lat_, b_[0], b_[1], b_[2], j_,
                                          alpha_, kSlDt);
                }
            }
            sim_time_ += n * kSlDt;
            display_changed_ = true;
        }
        refresh_bloch();
        rebuild_arrows();
        if (++frames_ % 10 == 0) {
            title_dirty_ = true;
        }
    }
    void tick() override {
        const int base = exact_mode_ ? kSlExactSteps : kSlStepsPerTick;
        const int per_tick = base * time_scale_;
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
        if (exact_mode_) {
            // Rotate every site so B_hat -> z, Born-measure the bits
            // sequentially (the correct joint sampling), rotate back.
            const double th = std::acos(std::clamp(nz, -1.0, 1.0));
            const double axn = std::hypot(-ny, nx);
            const double ax = axn > 1e-12 ? -ny / axn : 1.0;
            const double ay = axn > 1e-12 ? nx / axn : 0.0;
            for (int i = 0; i < kSlN * kSlN; ++i) {
                ses::exact_site_rotate(exact_, i, ax, ay, 0.0, -th);
            }
            for (int i = 0; i < kSlN * kSlN; ++i) {
                plus += ses::exact_measure_z(exact_, i, uni(rng_)) > 0
                            ? 1
                            : 0;
            }
            for (int i = 0; i < kSlN * kSlN; ++i) {
                ses::exact_site_rotate(exact_, i, ax, ay, 0.0, th);
            }
        } else {
            for (auto& s : lat_.s) {
                plus += ses::spin_measure(s, nx, ny, nz, uni(rng_)) > 0
                            ? 1
                            : 0;
            }
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
        if (key == 'X') {
            set_exact(!exact_mode_);
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

    std::string title_text() override {
        std::string s = strf(
            "16 interacting spins [%s]  |  t = %.1f au  J = %+.2f  "
            "alpha = %.2f%s  |M| = %.2f  Neel = %.2f  mean |<s>| = %.2f",
            exact_mode_ ? "EXACT 2^16 Heisenberg"
                        : "mean-field, no entanglement",
            sim_time_, j_, alpha_,
            exact_mode_ ? " (unused: closed system)" : "",
            magnetization(), staggered(), arrow_mean());
        if (!note_.empty()) {
            s += "  [" + note_ + "]";
        }
        s += "  keys: 2 random / 3 ferro / 4 Neel / X exact / M measure";
        return s;
    }

    // Glassy Bloch spheres, one per site (the fresnel marker pass).
    int marker_count() const override { return kSlN * kSlN; }
    SceneMarker marker(int i) const override {
        double cx = 0.0;
        double cy = 0.0;
        site_center(i, &cx, &cy);
        return {static_cast<float>(cx), static_cast<float>(cy), 0.0f,
                static_cast<float>(kSlR), 0.55f, 0.75f, 0.95f};
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
        if (exact_mode_) {
            exact_ = ses::exact_from_product(lat_);  // fresh product boot
        }
        refresh_bloch();
        sim_time_ = 0.0;
        pending_steps_ = 0;
        note_ = what;
        display_changed_ = true;
        title_dirty_ = true;
    }

    void refresh_bloch() {
        for (int i = 0; i < kSlN * kSlN; ++i) {
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            if (exact_mode_) {
                ses::exact_site_bloch(exact_, i, &x, &y, &z);
            } else {
                ses::bloch_vector(lat_.s[static_cast<std::size_t>(i)],
                                  &x, &y, &z);
            }
            bloch_[static_cast<std::size_t>(3 * i)] = x;
            bloch_[static_cast<std::size_t>(3 * i + 1)] = y;
            bloch_[static_cast<std::size_t>(3 * i + 2)] = z;
        }
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
            // Exact mode: |<sigma>| < 1 when entangled -- the arrow
            // honestly SHRINKS.
            const double x = bloch_[static_cast<std::size_t>(3 * site)];
            const double y =
                bloch_[static_cast<std::size_t>(3 * site + 1)];
            const double z =
                bloch_[static_cast<std::size_t>(3 * site + 2)];
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
    ses::SpinState16 exact_;  // the full 2^16 wavefunction (exact mode)
    bool exact_mode_ = false;
    // Per-site Bloch vectors (3 doubles/site), refreshed each frame from
    // whichever engine is live -- the single display source.
    std::vector<double> bloch_ =
        std::vector<double>(3 * kSlN * kSlN, 0.0);
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
