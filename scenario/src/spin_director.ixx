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
export module ses.scenario.spin_director;
export import ses.scenario;
export import ses.spin;


// One electron spin at origin: H = (1/2) B.sigma via exact SU(2) rotations
// (ses.spin -- no grid, no Trotter). E gives no torque on a pinned spin ->
// RED flux only.
// CONTRACT: tests/spin_test.cpp + --selftest-spin.
// volk.h textually first: VK_* macros never cross module boundaries.


export namespace ses_shell {

constexpr double kSpDt = 0.002;
constexpr int kSpStepsPerTick = 8;
constexpr double kSpR = 10.0;         // Bloch sphere radius
constexpr double kSpB0 = 0.5;         // boot |B| along z
constexpr double kSpRfB1 = 0.05;      // Omega_R
constexpr int kSpEcho = 64;           // echo ensemble size
constexpr double kSpEchoDet = 0.25;   // fractional detuning spread
constexpr double kSpEchoTau = 30.0;   // pulse spacing (au)
constexpr int kSpFlux = 30;           // tracer count per field
constexpr double kSpFluxR = 16.0;     // tracer shell radius
constexpr double kSpFluxSpeed = 0.12; // Bohr per frame

class SpinDirector final : public ScenarioDirector, public SpinApi {
public:
    SpinDirector() {
        rebuild_sphere();
        reset_state();
    }

    SpinApi* spin() override { return this; }

    // ---- SpinApi ----
    void set_b(int axis, double v) override {
        b_[axis] = std::clamp(v, -1.0, 1.0);
        title_dirty_ = true;
    }
    double b(int axis) const override { return b_[axis]; }
    void set_e(int axis, double v) override {
        e_[axis] = std::clamp(v, -1.0, 1.0);
        title_dirty_ = true;
    }
    double e(int axis) const override { return e_[axis]; }
    void toggle_rf() override {
        rf_on_ = !rf_on_;
        rf_w_ = bmag();      // resonance omega = |B|
        rf_t0_ = sim_time_;  // drive phase origin -> starts on +x
        title_dirty_ = true;
    }
    bool rf_on() const override { return rf_on_; }
    void pulse(bool half) override {
        for (ses::Spinor& s : ens_) {
            ses::spin_rotate(s, 1.0, 0.0, 0.0,
                             half ? 0.5 * kPi : kPi);
        }
        note_ = half ? "pi/2 pulse" : "pi pulse";
        title_dirty_ = true;
    }
    void spin_echo() override {
        ens_.assign(kSpEcho, ses::Spinor{});
        det_.resize(kSpEcho);
        std::uniform_real_distribution<double> d(-kSpEchoDet, kSpEchoDet);
        for (int i = 0; i < kSpEcho; ++i) {
            det_[static_cast<std::size_t>(i)] = 1.0 + d(rng_);
        }
        pulse(true);
        echo_stage_ = 1;
        echo_left_ = static_cast<int>(kSpEchoTau / kSpDt + 0.5);
        echo_peak_ = 0.0;
        note_ = "echo: dephasing...";
        title_dirty_ = true;
    }
    double echo_peak() const override { return echo_peak_; }
    double bloch_x() override { return mean_[0]; }
    double bloch_y() override { return mean_[1]; }
    double bloch_z() override { return mean_[2]; }
    int last_outcome() const override { return outcome_; }

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
            step_batch(n);
            display_changed_ = true;
        }
        advect_flux();
        rebuild_arrows();
        if (++frames_ % 10 == 0) {
            title_dirty_ = true;
        }
    }
    void tick() override {
        const int per_tick = kSpStepsPerTick * time_scale_;
        pending_steps_ = std::min(pending_steps_ + per_tick, per_tick);
    }

    void do_set_real_time() override {}
    void reset_simulation() override { reset_state(); }
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
        for (ses::Spinor& s : ens_) {
            plus += ses::spin_measure(s, nx, ny, nz, uni(rng_)) > 0 ? 1 : 0;
        }
        outcome_ = 2 * plus - static_cast<int>(ens_.size());
        note_ = strf("measured: %d of %d aligned with B", plus,
                     static_cast<int>(ens_.size()));
        echo_stage_ = 0;
        title_dirty_ = true;
    }
    void toggle_view_mode() override {}
    bool handle_key(char key) override {
        if (key == '2') {
            reset_state();
            return true;
        }
        if (key == '3') {
            pulse(true);
            return true;
        }
        if (key == '4') {
            pulse(false);
            return true;
        }
        if (key == '5') {
            spin_echo();
            return true;
        }
        if (key == 'L') {
            toggle_rf();
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
    double sim_dt() const override { return kSpDt; }
    int steps_per_tick_x1() const override { return kSpStepsPerTick; }

    // ---- display (overlay-only scene) ----
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
    // A glassy Bloch sphere surface at the origin (same style as the lattice
    // sites), so the single spin reads as a sphere, not bare wireframe rings.
    int marker_count() const override { return 1; }
    SceneMarker marker(int /*i*/) const override {
        return {0.0f, 0.0f, 0.0f, static_cast<float>(kSpR),
                0.55f, 0.75f, 0.95f};
    }

    std::string title_text() override {
        std::string s = strf(
            "Electron spin (Bloch sphere)  |  t = %.1f au  |B| = %.2f "
            "(omega_L = |B|)  <s> = (%+.2f, %+.2f, %+.2f)  spins %d",
            sim_time_, bmag(), mean_[0], mean_[1], mean_[2],
            static_cast<int>(ens_.size()));
        if (rf_on_) {
            s += strf("  RF ON (Omega_R = %.2f)", kSpRfB1);
        }
        if (std::abs(e_[0]) + std::abs(e_[1]) + std::abs(e_[2]) > 1e-9) {
            s += "  [E: global phase only on a pinned spin]";
        }
        if (!note_.empty()) {
            s += "  [" + note_ + "]";
        }
        s += "  keys: 2 reset / 3 pi/2 / 4 pi / 5 echo / L RF / M measure";
        return s;
    }

    // SEPARATE curves: a line strip draws bridges between points, so chaining
    // rings would smear chords across the sphere.
    int overlay_curve_count() const override {
        return 8 + static_cast<int>(e_flux_.size() / 6) +
               static_cast<int>(b_flux_.size() / 6);
    }
    OverlayCurve overlay_curve(int i) const override {
        if (i < 3) {
            return {&circle_[static_cast<std::size_t>(i)][0],
                    static_cast<int>(circle_[static_cast<std::size_t>(i)]
                                         .size() /
                                     3),
                    0.75f, 0.80f, 0.95f, 0.30f};
        }
        if (i < 6) {
            const int a = i - 3;
            static constexpr float kAxisRgb[3][3] = {
                {0.95f, 0.40f, 0.40f},
                {0.40f, 0.95f, 0.40f},
                {0.45f, 0.60f, 1.00f}};
            return {&axis_[static_cast<std::size_t>(a)][0], 2,
                    kAxisRgb[a][0], kAxisRgb[a][1], kAxisRgb[a][2], 0.6f};
        }
        if (i == 6) {
            return {fan_.data(), static_cast<int>(fan_.size() / 3),
                    0.75f, 0.95f, 0.85f, 0.22f};
        }
        if (i == 7) {
            return {arrow_.data(), static_cast<int>(arrow_.size() / 3),
                    1.0f, 0.75f, 0.20f, 1.0f};
        }
        const int k = i - 8;
        const int ne = static_cast<int>(e_flux_.size() / 6);
        if (k < ne) {
            return {&e_flux_[static_cast<std::size_t>(6 * k)], 2,
                    1.0f, 0.25f, 0.20f, 0.8f};
        }
        return {&b_flux_[static_cast<std::size_t>(6 * (k - ne))], 2,
                0.25f, 1.0f, 0.35f, 0.8f};
    }

    double default_camera_azimuth() const override { return 0.6; }
    double default_camera_elevation() const override { return 0.35; }
    double default_camera_distance() const override { return 55.0; }

private:
    static constexpr double kPi = 3.14159265358979323846;

    double bmag() const {
        return std::sqrt(b_[0] * b_[0] + b_[1] * b_[1] + b_[2] * b_[2]);
    }

    void reset_state() {
        ens_.assign(1, ses::Spinor{});
        // Boot on +x: the precession is visible immediately.
        ses::spin_rotate(ens_[0], 0.0, 1.0, 0.0, 0.5 * kPi);
        det_.assign(1, 1.0);
        echo_stage_ = 0;
        echo_peak_ = 0.0;
        outcome_ = 0;
        sim_time_ = 0.0;
        pending_steps_ = 0;
        note_.clear();
        display_changed_ = true;
        title_dirty_ = true;
    }

    void step_batch(int n) {
        for (int k = 0; k < n; ++k) {
            const double t = sim_time_ + k * kSpDt;
            double bx = b_[0];
            double by = b_[1];
            double bz = b_[2];
            if (rf_on_) {
                bx += kSpRfB1 * std::cos(rf_w_ * (t - rf_t0_));
                by += kSpRfB1 * std::sin(rf_w_ * (t - rf_t0_));
            }
            for (std::size_t i = 0; i < ens_.size(); ++i) {
                const double d = det_[i];
                ses::spin_step(ens_[i], bx * d, by * d, bz * d, kSpDt);
            }
            if (echo_stage_ != 0 && --echo_left_ <= 0) {
                if (echo_stage_ == 1) {
                    pulse(false);  // the refocusing pi
                    echo_stage_ = 2;
                    echo_left_ =
                        static_cast<int>(kSpEchoTau / kSpDt + 0.5);
                    note_ = "echo: refocusing...";
                } else {
                    measure_mean();
                    echo_peak_ = std::hypot(mean_[0], mean_[1]);
                    echo_stage_ = 0;
                    note_ = strf("ECHO |m_xy| = %.2f", echo_peak_);
                }
                title_dirty_ = true;
            }
        }
        sim_time_ += n * kSpDt;
        measure_mean();
    }

    void measure_mean() {
        double sx = 0.0;
        double sy = 0.0;
        double sz = 0.0;
        for (const ses::Spinor& s : ens_) {
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            ses::bloch_vector(s, &x, &y, &z);
            sx += x;
            sy += y;
            sz += z;
        }
        const double inv = 1.0 / static_cast<double>(ens_.size());
        mean_[0] = sx * inv;
        mean_[1] = sy * inv;
        mean_[2] = sz * inv;
    }

    void rebuild_sphere() {
        // great circle about each axis (circle about X spans y-z)
        for (int axis = 0; axis < 3; ++axis) {
            std::vector<float>& c = circle_[static_cast<std::size_t>(axis)];
            c.clear();
            for (int t = 0; t <= 48; ++t) {
                const double th = 2.0 * kPi * t / 48.0;
                float p[3] = {0.0f, 0.0f, 0.0f};
                p[(axis + 1) % 3] =
                    static_cast<float>(kSpR * std::cos(th));
                p[(axis + 2) % 3] =
                    static_cast<float>(kSpR * std::sin(th));
                c.push_back(p[0]);
                c.push_back(p[1]);
                c.push_back(p[2]);
            }
        }
        // axis lines poking past the sphere
        for (int a = 0; a < 3; ++a) {
            std::vector<float>& ax = axis_[static_cast<std::size_t>(a)];
            ax.assign(6, 0.0f);
            ax[static_cast<std::size_t>(a)] =
                static_cast<float>(-1.15 * kSpR);
            ax[static_cast<std::size_t>(3 + a)] =
                static_cast<float>(1.15 * kSpR);
        }
    }

    void rebuild_arrows() {
        fan_.clear();
        if (ens_.size() > 1) {
            for (const ses::Spinor& s : ens_) {
                double x = 0.0;
                double y = 0.0;
                double z = 0.0;
                ses::bloch_vector(s, &x, &y, &z);
                if (!fan_.empty()) {
                    const std::size_t nn = fan_.size();
                    fan_.push_back(fan_[nn - 3]);
                    fan_.push_back(fan_[nn - 2]);
                    fan_.push_back(fan_[nn - 1]);
                    fan_.push_back(0.0f);
                    fan_.push_back(0.0f);
                    fan_.push_back(0.0f);
                }
                fan_.push_back(0.0f);
                fan_.push_back(0.0f);
                fan_.push_back(0.0f);
                fan_.push_back(static_cast<float>(kSpR * x));
                fan_.push_back(static_cast<float>(kSpR * y));
                fan_.push_back(static_cast<float>(kSpR * z));
            }
        }
        arrow_.clear();
        const double ax = kSpR * mean_[0];
        const double ay = kSpR * mean_[1];
        const double az = kSpR * mean_[2];
        arrow_.push_back(0.0f);
        arrow_.push_back(0.0f);
        arrow_.push_back(0.0f);
        arrow_.push_back(static_cast<float>(ax));
        arrow_.push_back(static_cast<float>(ay));
        arrow_.push_back(static_cast<float>(az));
        const double len = std::sqrt(ax * ax + ay * ay + az * az);
        if (len > 1e-6) {
            const double hx = ax * (1.0 - 1.5 / len);
            const double hy = ay * (1.0 - 1.5 / len);
            const double hz = az * (1.0 - 1.5 / len);
            // pick any transverse direction for the head wings
            double px = -ay;
            double py = ax;
            double pz = 0.0;
            double pn = std::sqrt(px * px + py * py + pz * pz);
            if (pn < 1e-9) {
                px = 1.0;
                py = 0.0;
                pn = 1.0;
            }
            px *= 0.8 / pn;
            py *= 0.8 / pn;
            pz *= 0.8 / pn;
            for (int w = -1; w <= 1; w += 2) {
                arrow_.push_back(static_cast<float>(ax));
                arrow_.push_back(static_cast<float>(ay));
                arrow_.push_back(static_cast<float>(az));
                arrow_.push_back(static_cast<float>(hx + w * px));
                arrow_.push_back(static_cast<float>(hy + w * py));
                arrow_.push_back(static_cast<float>(hz + w * pz));
            }
        }
    }

    // Lagrangian field tracers
    void advect_flux() {
        advect_one(e_flux_, e_pos_, e_[0], e_[1], e_[2]);
        advect_one(b_flux_, b_pos_, b_[0], b_[1], b_[2]);
        display_changed_ = true;
    }
    void advect_one(std::vector<float>& out, std::vector<double>& pos,
                    double fx, double fy, double fz) {
        out.clear();
        const double f = std::sqrt(fx * fx + fy * fy + fz * fz);
        if (f < 1e-9) {
            pos.clear();
            return;
        }
        const double dx = fx / f;
        const double dy = fy / f;
        const double dz = fz / f;
        std::uniform_real_distribution<double> uni(-kSpFluxR, kSpFluxR);
        if (pos.size() != 3 * kSpFlux) {
            pos.resize(3 * kSpFlux);
            for (std::size_t i = 0; i < pos.size(); ++i) {
                pos[i] = uni(rng_);
            }
        }
        const double sp = kSpFluxSpeed * std::min(1.0, f);
        for (int i = 0; i < kSpFlux; ++i) {
            double& x = pos[3 * i];
            double& y = pos[3 * i + 1];
            double& z = pos[3 * i + 2];
            x += dx * sp;
            y += dy * sp;
            z += dz * sp;
            if (x * x + y * y + z * z > kSpFluxR * kSpFluxR * 1.4) {
                x = uni(rng_) * 0.8 - dx * kSpFluxR;
                y = uni(rng_) * 0.8 - dy * kSpFluxR;
                z = uni(rng_) * 0.8 - dz * kSpFluxR;
            }
            // separate 2-point streak per tracer (chaining would smear)
            const double tl = 1.6 * std::min(1.0, f);
            out.push_back(static_cast<float>(x));
            out.push_back(static_cast<float>(y));
            out.push_back(static_cast<float>(z));
            out.push_back(static_cast<float>(x - dx * tl));
            out.push_back(static_cast<float>(y - dy * tl));
            out.push_back(static_cast<float>(z - dz * tl));
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

    ses::Grid3D grid_{ses::Grid1D{-20.0, 20.0, 2},
                      ses::Grid1D{-20.0, 20.0, 2},
                      ses::Grid1D{-20.0, 20.0, 2}};
    std::vector<ses::Spinor> ens_;
    std::vector<double> det_;
    double b_[3] = {0.0, 0.0, kSpB0};
    double e_[3] = {0.0, 0.0, 0.0};
    double mean_[3] = {1.0, 0.0, 0.0};
    bool rf_on_ = false;
    double rf_w_ = kSpB0;
    double rf_t0_ = 0.0;
    int echo_stage_ = 0;
    int echo_left_ = 0;
    double echo_peak_ = 0.0;
    int outcome_ = 0;
    double sim_time_ = 0.0;
    int pending_steps_ = 0;
    int time_scale_ = 1;
    std::uint64_t frames_ = 0;
    bool display_changed_ = true;
    bool title_dirty_ = true;
    bool compute_attempted_ = false;
    std::string note_;
    std::mt19937 rng_{20260721u};
    std::vector<float> circle_[3];
    std::vector<float> axis_[3];
    std::vector<float> fan_;
    std::vector<float> arrow_;
    std::vector<float> e_flux_;
    std::vector<float> b_flux_;
    std::vector<double> e_pos_;
    std::vector<double> b_pos_;
    ses::Mesh no_mesh_;
    std::vector<ses::Rgb> no_colors_;
    std::vector<float> no_staging_;
};

}  // namespace ses_shell
