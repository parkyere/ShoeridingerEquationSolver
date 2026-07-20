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
export module ses.scenario.doubleslit2d_director;
export import ses.scenario;
export import ses.field;
export import ses.grid;
export import ses.lattice2d;
export import ses.potential;
import ses.parallel;
import ses.heightfield;


// The REAL double-slit + Aharonov-Bohm experiment, in 2D, literally: an
// electron packet flies +x into a high potential wall pierced by two
// slits; a solenoid (flux along z, drawn as the amber arrow + core circle)
// hides INSIDE the wall between the slits; a screen line on the right
// accumulates the arrival density over time. Physics is the Peierls
// lattice propagator (ses.lattice2d): the flux enters as EXACT link
// phases, B = 0 in every plaquette the electron can reach -- pure AB
// fringe shift, period 2 pi.
//
// The 2D plane is displayed through the volume path: physics lives on one
// z-plane (nz = 1), the staging replicates it into a thin display slab, and
// the standard phase-hued cloud + Z-snap gives the face-on view.
// volk.h textually first: VK_* macros never cross module boundaries.


export namespace ses_shell {

constexpr double kDs2dBoxX = 60.0;
constexpr double kDs2dBoxY = 60.0;
constexpr int kDs2dNx = 512;
constexpr int kDs2dNy = 512;
constexpr int kDs2dNz = 4;        // display slab thickness (cells)
constexpr double kDs2dZHalf = 2.0;
constexpr double kDs2dDt = 0.01;
constexpr double kDs2dK0 = 2.0;   // +x mechanical momentum (kh ~ 0.47)
constexpr double kDs2dSigma = 8.0;
constexpr double kDs2dLaunchX = -35.0;
constexpr double kDs2dWallLo = 0.0;
constexpr double kDs2dWallHi = 1.5;
constexpr double kDs2dWallV = 40.0;  // kappa*w ~ 13: leak e^-13, honest wall
constexpr double kDs2dSep = 8.0;
constexpr double kDs2dSepMin = 4.0;
constexpr double kDs2dSepMax = 16.0;
constexpr double kDs2dWidth = 2.0;
constexpr double kDs2dWidthMin = 1.0;
constexpr double kDs2dWidthMax = 4.0;
constexpr double kDs2dScreenX = 45.0;
constexpr double kDs2dAbsorb = 10.0;
constexpr int kDs2dStepsPerTick = 10;  // ~6 au/s at 60 fps: 13 s transit
// Continuous electron beam (user order): a coherent ON-SHELL source feeds
// the box every step -- psi += A src e^{-i w t} dt, w = the lattice band
// energy of k0 -- against the open edge absorbers; injection balances
// absorption into a steady interference state.
// CONTRACT: lattice2d_test BeamSourceWithOpenBoundaryReachesSteadyState.
constexpr double kDs2dSrcAmp = 0.05;
constexpr double kDs2dSrcSigX = 2.0;
// IBM-style STM height surface (like the corral): z = |psi|^2 peak-tracked.
constexpr double kDs2dSurfH = 6.0;
constexpr int kDs2dMeshStride = 1;  // 512^2 physics = 512^2 display mesh

class DoubleSlit2DDirector final : public ScenarioDirector, public SlitApi {
public:
    DoubleSlit2DDirector()
        : phys_grid_{ses::Grid1D{-kDs2dBoxX, kDs2dBoxX, kDs2dNx},
                     ses::Grid1D{-kDs2dBoxY, kDs2dBoxY, kDs2dNy},
                     ses::Grid1D{-1.0, 1.0, 1}},
          disp_grid_{ses::Grid1D{-kDs2dBoxX, kDs2dBoxX, kDs2dNx},
                     ses::Grid1D{-kDs2dBoxY, kDs2dBoxY, kDs2dNy},
                     ses::Grid1D{-kDs2dZHalf, kDs2dZHalf, kDs2dNz}},
          psi_{phys_grid_},
          src_{phys_grid_} {
        build_mask();
        rebuild_wall_and_prop();
        fire();
    }

    SlitApi* slit() override { return this; }

    // ---- SlitApi ----
    void set_separation(double d) override {
        sep_ = std::clamp(d, kDs2dSepMin, kDs2dSepMax);
        rebuild_wall_and_prop();
        fire();
    }
    double separation() const override { return sep_; }
    void set_width(double w) override {
        width_ = std::clamp(w, kDs2dWidthMin, kDs2dWidthMax);
        rebuild_wall_and_prop();
        fire();
    }
    double width() const override { return width_; }
    void set_flux(double phi) override {
        flux_ = phi;
        prop_->set_solenoid(flux_, solenoid_x(), 0.0);
        fire();
        rebuild_props_overlays();  // the flux arrow length encodes Phi
    }
    double flux() const override { return flux_; }
    void refire() override { fire(); }
    // Instantaneous probability past the wall (the absorbers eat both
    // exits eventually; read while the pattern crosses the screen).
    double transmitted_fraction() const override {
        const double cell = phys_grid_.x.spacing() * phys_grid_.y.spacing() *
                            phys_grid_.z.spacing();
        double acc = 0.0;
        for (int j = 0; j < kDs2dNy; ++j) {
            for (int i = 0; i < kDs2dNx; ++i) {
                if (phys_grid_.x.coord(i) > kDs2dWallHi) {
                    acc += std::norm(psi_(i, j, 0));
                }
            }
        }
        return acc * cell;
    }
    // The accumulated screen histogram at the row nearest y.
    double screen_at(double y) const override {
        int best = 0;
        for (int j = 1; j < kDs2dNy; ++j) {
            if (std::abs(phys_grid_.y.coord(j) - y) <
                std::abs(phys_grid_.y.coord(best) - y)) {
                best = j;
            }
        }
        return screen_[static_cast<std::size_t>(best)];
    }

    // ---- lifecycle ----
    const ses::Grid3D& grid() const override { return disp_grid_; }
    void init_compute(ses_vk::DeviceContext& /*ctx*/, bool /*device_ok*/,
                      std::int64_t /*free_vram*/) override {
        compute_attempted_ = true;  // CPU lattice: display via staging
    }
    void release_gpu() override {}
    bool compute_attempted() const override { return compute_attempted_; }
    bool gpu_ok() const override { return false; }

    // ---- per-frame ----
    void run_frame() override {
        if (pending_steps_ > 0) {
            const int n = pending_steps_;
            pending_steps_ = 0;
            step_batch(n);
            sim_time_ += n * kDs2dDt;
            display_changed_ = true;
            vol_dirty_ = true;
            staging_dirty_ = true;
        }
        if (staging_dirty_) {
            staging_dirty_ = false;
            // Peak SNAP then 0.98-decay (the corral rule).
            double cur = 0.0;
            for (int j = 0; j < kDs2dNy; ++j) {
                for (int i = 0; i < kDs2dNx; ++i) {
                    cur = std::max(cur, std::norm(psi_(i, j, 0)));
                }
            }
            disp_peak_ = disp_peak_ <= 0.0 ? cur
                                           : std::max(cur, 0.98 * disp_peak_);
            if (disp_peak_ > 0.0) {
                hf_ = ses::heightfield_surface(psi_, kDs2dSurfH, disp_peak_,
                                               kDs2dMeshStride);
                mesh_dirty_ = true;
            }
            rebuild_screen_curve();
        }
        if (++frames_ % 10 == 0) {
            title_dirty_ = true;
        }
    }
    void tick() override {
        const int per_tick = kDs2dStepsPerTick * time_scale_;
        pending_steps_ = std::min(pending_steps_ + per_tick, per_tick);
    }

    // ---- controls ----
    void do_set_real_time() override {}
    void reset_simulation() override { fire(); }
    void measure_now() override {}
    void toggle_view_mode() override {}  // cloud-only scene
    bool handle_key(char key) override {
        if (key == '2') {
            fire();
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
    double sim_dt() const override { return kDs2dDt; }
    int steps_per_tick_x1() const override { return kDs2dStepsPerTick; }

    // ---- display ----
    bool cloud() const override { return false; }  // STM height surface
    double peak() const override { return peak_; }
    VkImageView psi_volume_view() override { return VK_NULL_HANDLE; }
    float next_flash_intensity() override { return 0.0f; }
    bool take_volume_written() override {
        return std::exchange(display_changed_, false);
    }
    bool take_volume_dirty() override {
        return std::exchange(vol_dirty_, false);
    }
    bool take_mesh_dirty() override {
        return std::exchange(mesh_dirty_, false);
    }
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
    const ses::Mesh& mesh() const override { return hf_.mesh; }
    const std::vector<ses::Rgb>& colors() const override {
        return hf_.colors;
    }
    std::string title_text() override {
        const double pi = 3.14159265358979323846;
        return strf(
            "Electron double slit + Aharonov-Bohm (2D lattice)  |  t = %.1f "
            "au (%dx%d, dt %.2g)  d = %.1f  w = %.1f  Phi = %.2f pi  "
            "T = %.1f%%  B = 0 on every electron path  keys: 2 refire",
            sim_time_, kDs2dNx, kDs2dNy, kDs2dDt, sep_, width_, flux_ / pi,
            100.0 * transmitted_fraction());
    }

    int marker_count() const override { return 0; }

    // Overlays: the wall slabs (filled), the solenoid arrow + core circle,
    // the screen plate, and the accumulated arrival histogram.
    int overlay_curve_count() const override { return 4; }
    OverlayCurve overlay_curve(int i) const override {
        if (i == 0) {  // pierced wall: translucent red slabs
            return {wall_curve_.data(),
                    static_cast<int>(wall_curve_.size() / 3),
                    1.0f, 0.30f, 0.25f, 0.55f, true};
        }
        if (i == 1) {  // solenoid: amber flux arrow along z + core circle
            return {solenoid_curve_.data(),
                    static_cast<int>(solenoid_curve_.size() / 3),
                    1.0f, 0.70f, 0.20f, 1.0f};
        }
        if (i == 2) {  // screen plate
            return {plate_curve_.data(),
                    static_cast<int>(plate_curve_.size() / 3),
                    0.85f, 0.85f, 0.90f, 0.35f};
        }
        // accumulated arrivals: cyan histogram growing off the plate
        return {screen_curve_.data(),
                static_cast<int>(screen_curve_.size() / 3),
                0.35f, 0.85f, 1.0f, 0.95f};
    }

    double default_camera_azimuth() const override { return 0.35; }
    double default_camera_elevation() const override { return 0.95; }
    double default_camera_distance() const override { return 170.0; }

private:
    double solenoid_x() const {
        // Buried mid-wall, quarter-cell off a lattice line so the cell
        // containing it is unambiguous.
        return 0.5 * (kDs2dWallLo + kDs2dWallHi) +
               0.25 * phys_grid_.x.spacing();
    }

    void build_mask() {
        // 2D absorber frame (x and y edges) built from the 1D profiles.
        const std::vector<double> mx =
            ses::absorbing_mask(phys_grid_.x, kDs2dAbsorb);
        const std::vector<double> my =
            ses::absorbing_mask(phys_grid_.y, kDs2dAbsorb);
        mask_.resize(static_cast<std::size_t>(kDs2dNx * kDs2dNy));
        for (int j = 0; j < kDs2dNy; ++j) {
            for (int i = 0; i < kDs2dNx; ++i) {
                mask_[static_cast<std::size_t>(j * kDs2dNx + i)] =
                    mx[static_cast<std::size_t>(i)] *
                    my[static_cast<std::size_t>(j)];
            }
        }
    }

    bool slit_open(double y) const {
        return std::abs(y - 0.5 * sep_) <= 0.5 * width_ ||
               std::abs(y + 0.5 * sep_) <= 0.5 * width_;
    }

    void rebuild_wall_and_prop() {
        std::vector<double> v(static_cast<std::size_t>(phys_grid_.size()),
                              0.0);
        for (int j = 0; j < kDs2dNy; ++j) {
            const double y = phys_grid_.y.coord(j);
            if (slit_open(y)) {
                continue;
            }
            for (int i = 0; i < kDs2dNx; ++i) {
                const double x = phys_grid_.x.coord(i);
                if (x >= kDs2dWallLo && x <= kDs2dWallHi) {
                    v[static_cast<std::size_t>(phys_grid_.flat(i, j, 0))] =
                        kDs2dWallV;
                }
            }
        }
        prop_ = std::make_unique<ses::PeierlsLattice2D>(phys_grid_, v,
                                                        kDs2dDt);
        prop_->set_solenoid(flux_, solenoid_x(), 0.0);
        rebuild_props_overlays();
    }

    void fire() {
        // Vacuum boot: the CONTINUOUS beam fills the box (no one-shot
        // packet, no normalization -- the steady state balances the source
        // against the edge absorbers). The emitter is a narrow column at
        // the old launch x, wide in y, carrying the +x on-shell phase.
        ses::parallel_for(kDs2dNy, [&](int j) {
            const double y = phys_grid_.y.coord(j);
            for (int i = 0; i < kDs2dNx; ++i) {
                const double x = phys_grid_.x.coord(i);
                const double dx = x - kDs2dLaunchX;
                psi_(i, j, 0) = 0.0;
                src_(i, j, 0) =
                    std::exp(-dx * dx / (2.0 * kDs2dSrcSigX * kDs2dSrcSigX) -
                             y * y / (4.0 * kDs2dSigma * kDs2dSigma)) *
                    std::complex<double>{std::cos(kDs2dK0 * x),
                                         std::sin(kDs2dK0 * x)};
            }
        });
        const double hx = phys_grid_.x.spacing();
        src_omega_ = (1.0 - std::cos(kDs2dK0 * hx)) / (hx * hx);
        disp_peak_ = 0.0;
        peak_ = 1.0;
        screen_.assign(static_cast<std::size_t>(kDs2dNy), 0.0);
        sim_time_ = 0.0;
        pending_steps_ = 0;
        display_changed_ = true;
        vol_dirty_ = true;
        staging_dirty_ = true;
        title_dirty_ = true;
    }

    void step_batch(int n) {
        int i_scr = 0;
        for (int i = 1; i < kDs2dNx; ++i) {
            if (std::abs(phys_grid_.x.coord(i) - kDs2dScreenX) <
                std::abs(phys_grid_.x.coord(i_scr) - kDs2dScreenX)) {
                i_scr = i;
            }
        }
        for (int s = 0; s < n; ++s) {
            prop_->step(psi_);
            // Edge absorbers (leaked flux VANISHES -- open stage) + the
            // continuous on-shell source injection.
            sim_beam_t_ += kDs2dDt;
            const std::complex<double> ph{std::cos(src_omega_ * sim_beam_t_),
                                          -std::sin(src_omega_ * sim_beam_t_)};
            ses::parallel_for(kDs2dNy, [&](int j) {
                const std::size_t row =
                    static_cast<std::size_t>(j) *
                    static_cast<std::size_t>(kDs2dNx);
                for (int i = 0; i < kDs2dNx; ++i) {
                    const std::size_t c = row + static_cast<std::size_t>(i);
                    psi_.data()[c] =
                        psi_.data()[c] *
                            mask_[static_cast<std::size_t>(j * kDs2dNx + i)] +
                        kDs2dSrcAmp * kDs2dDt * ph * src_.data()[c];
                }
            });
            // The screen integrates arrivals: sum |psi|^2 dt on its line.
            for (int j = 0; j < kDs2dNy; ++j) {
                screen_[static_cast<std::size_t>(j)] +=
                    std::norm(psi_(i_scr, j, 0)) * kDs2dDt;
            }
        }
        // Brightness normalizer: decayed running max (no flicker as the
        // packet dilutes through the slits).
        double pk = 0.0;
        for (int j = 0; j < kDs2dNy; ++j) {
            for (int i = 0; i < kDs2dNx; ++i) {
                pk = std::max(pk, std::norm(psi_(i, j, 0)));
            }
        }
        peak_ = std::max(pk, 0.98 * peak_);
    }

    void rebuild_staging() {
        const std::size_t plane =
            static_cast<std::size_t>(kDs2dNx) *
            static_cast<std::size_t>(kDs2dNy);
        staging_.resize(plane * kDs2dNz * 2);
        ses::parallel_for(kDs2dNy, [&](int j) {
            for (int i = 0; i < kDs2dNx; ++i) {
                const std::complex<double> z = psi_(i, j, 0);
                const std::size_t o =
                    2 * (static_cast<std::size_t>(j) *
                             static_cast<std::size_t>(kDs2dNx) +
                         static_cast<std::size_t>(i));
                staging_[o] = static_cast<float>(z.real());
                staging_[o + 1] = static_cast<float>(z.imag());
            }
        });
        // Replicate the plane through the display slab.
        for (int k = 1; k < kDs2dNz; ++k) {
            std::copy(staging_.begin(),
                      staging_.begin() + static_cast<std::ptrdiff_t>(
                                             plane * 2),
                      staging_.begin() +
                          static_cast<std::ptrdiff_t>(plane * 2) * k);
        }
    }

    // Static scene furniture: wall slabs, solenoid arrow + core circle,
    // screen plate.
    void rebuild_props_overlays() {
        wall_curve_.clear();
        auto slab = [&](double y0, double y1) {
            if (y1 <= y0) {
                return;
            }
            const float x0 = static_cast<float>(kDs2dWallLo);
            const float x1 = static_cast<float>(kDs2dWallHi);
            const float a = static_cast<float>(y0);
            const float b = static_cast<float>(y1);
            const float quad[12] = {x0, a, 0.0f, x1, a, 0.0f,
                                    x0, b, 0.0f, x1, b, 0.0f};
            if (!wall_curve_.empty()) {
                // degenerate bridge: repeat last vertex + this quad's head
                const std::size_t n = wall_curve_.size();
                wall_curve_.push_back(wall_curve_[n - 3]);
                wall_curve_.push_back(wall_curve_[n - 2]);
                wall_curve_.push_back(wall_curve_[n - 1]);
                wall_curve_.push_back(quad[0]);
                wall_curve_.push_back(quad[1]);
                wall_curve_.push_back(quad[2]);
            }
            wall_curve_.insert(wall_curve_.end(), quad, quad + 12);
        };
        slab(-kDs2dBoxY, -0.5 * sep_ - 0.5 * width_);
        slab(-0.5 * sep_ + 0.5 * width_, 0.5 * sep_ - 0.5 * width_);
        slab(0.5 * sep_ + 0.5 * width_, kDs2dBoxY);

        // Solenoid: core circle in the plane + flux arrow along z with
        // signed length ~ Phi (zero-length = invisible at Phi = 0).
        solenoid_curve_.clear();
        const double pi = 3.14159265358979323846;
        const float xs = static_cast<float>(solenoid_x());
        auto put = [&](float x, float y, float z) {
            solenoid_curve_.push_back(x);
            solenoid_curve_.push_back(y);
            solenoid_curve_.push_back(z);
        };
        const float rc = 1.2f;
        for (int t = 0; t <= 24; ++t) {
            const double th = 2.0 * pi * t / 24.0;
            put(xs + rc * static_cast<float>(std::cos(th)),
                rc * static_cast<float>(std::sin(th)), 0.0f);
        }
        const float len =
            static_cast<float>(5.0 * flux_ / (2.0 * pi));
        put(xs, 0.0f, 0.0f);
        put(xs, 0.0f, len);
        const float back = len - 0.18f * len - std::copysign(0.6f, len);
        put(xs - 0.8f, 0.0f, back);
        put(xs, 0.0f, len);
        put(xs + 0.8f, 0.0f, back);

        // Screen plate.
        plate_curve_ = {static_cast<float>(kDs2dScreenX),
                        -static_cast<float>(kDs2dBoxY - kDs2dAbsorb), 0.0f,
                        static_cast<float>(kDs2dScreenX),
                        static_cast<float>(kDs2dBoxY - kDs2dAbsorb), 0.0f};
    }

    void rebuild_screen_curve() {
        double smax = 0.0;
        for (const double s : screen_) {
            smax = std::max(smax, s);
        }
        // Absolute floor: the histogram is drawn self-normalized, and
        // before any real arrival the accumulated Gaussian TAIL (~1e-22)
        // would otherwise render as a full-height fake pattern.
        const double scale = smax > 1e-6 ? 12.0 / smax : 0.0;
        screen_curve_.clear();
        for (int j = 0; j < kDs2dNy; ++j) {
            const double y = phys_grid_.y.coord(j);
            if (std::abs(y) > kDs2dBoxY - kDs2dAbsorb) {
                continue;
            }
            screen_curve_.push_back(static_cast<float>(
                kDs2dScreenX + scale * screen_[static_cast<std::size_t>(j)]));
            screen_curve_.push_back(static_cast<float>(y));
            screen_curve_.push_back(0.0f);
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
    std::unique_ptr<ses::PeierlsLattice2D> prop_;
    std::vector<double> mask_;
    std::vector<double> screen_;
    std::vector<float> staging_;
    std::vector<float> wall_curve_;
    std::vector<float> solenoid_curve_;
    std::vector<float> plate_curve_;
    std::vector<float> screen_curve_;
    double sep_ = kDs2dSep;
    double width_ = kDs2dWidth;
    double flux_ = 0.0;
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
    ses::Field3D src_;
    double src_omega_ = 0.0;
    double sim_beam_t_ = 0.0;
    ses::Heightfield hf_;
    bool mesh_dirty_ = false;
    double disp_peak_ = 0.0;
    std::vector<ses::Rgb> no_colors_;
};

}  // namespace ses_shell
