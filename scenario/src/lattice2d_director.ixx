module;
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
export module ses.scenario.lattice2d_director;
export import ses.scenario;
export import ses.field;
export import ses.grid;
export import ses.lattice2d;
import ses.parallel;


// Shared base for the CPU 2D lattice scenes (double slit, Landau,
// corral, quantum dot): physics on one z-plane (nz = 1) of a Field3D,
// displayed through the volume path by replicating the plane into a thin
// z slab (CPU staging fallback -- gpu_ok() false by design). Subclasses
// own the propagator and stepping (do_steps), the base owns the frame
// plumbing, staging, and display flags.
// volk.h textually first: VK_* macros never cross module boundaries.


export namespace ses_shell {

class Lattice2DDirectorBase : public ScenarioDirector {
public:
    // ---- lifecycle ----
    const ses::Grid3D& grid() const override { return disp_grid_; }
    void init_compute(ses_vk::DeviceContext& /*ctx*/, bool /*device_ok*/,
                      std::int64_t /*free_vram*/) override {
        compute_attempted_ = true;
    }
    void release_gpu() override {}
    bool compute_attempted() const override { return compute_attempted_; }
    bool gpu_ok() const override { return false; }

    // ---- per-frame ----
    void run_frame() override {
        if (pending_steps_ > 0) {
            const int n = pending_steps_;
            pending_steps_ = 0;
            do_steps(n);
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
        const int per_tick = steps_per_tick() * time_scale_;
        pending_steps_ = std::min(pending_steps_ + per_tick, per_tick);
    }

    // ---- controls ----
    void set_real_time() override {}
    void measure_now() override {}
    void toggle_view_mode() override {}  // cloud-only scenes

    bool solving() const override { return false; }
    bool scene_ready() const override { return true; }
    void set_time_scale(int scale) override {
        time_scale_ = std::clamp(scale, 1, 16);
    }
    int time_scale() const override { return time_scale_; }
    double sim_time() const override { return sim_time_; }

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

    int marker_count() const override { return 0; }

    double default_camera_azimuth() const override { return 0.0; }
    double default_camera_elevation() const override { return 0.0; }

protected:
    Lattice2DDirectorBase(double box, int n, int nz, double z_half)
        : phys_grid_{ses::Grid1D{-box, box, n}, ses::Grid1D{-box, box, n},
                     ses::Grid1D{-1.0, 1.0, 1}},
          disp_grid_{ses::Grid1D{-box, box, n}, ses::Grid1D{-box, box, n},
                     ses::Grid1D{-z_half, z_half, nz}},
          nz_(nz),
          psi_{phys_grid_} {}

    // Scene hooks: the subclass owns the propagator and the batch.
    virtual void do_steps(int n) = 0;
    virtual int steps_per_tick() const { return 8; }

    // Reset the clock and the display flags around a re-preparation.
    void mark_fired() {
        sim_time_ = 0.0;
        pending_steps_ = 0;
        display_changed_ = true;
        vol_dirty_ = true;
        staging_dirty_ = true;
        title_dirty_ = true;
    }

    void track_peak() {
        const int n = phys_grid_.x.n;
        double pk = 0.0;
        for (int j = 0; j < phys_grid_.y.n; ++j) {
            for (int i = 0; i < n; ++i) {
                pk = std::max(pk, std::norm(psi_(i, j, 0)));
            }
        }
        peak_ = std::max(pk, 0.98 * peak_);
    }

    void rebuild_staging() {
        const int nx = phys_grid_.x.n;
        const int ny = phys_grid_.y.n;
        const std::size_t plane =
            static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
        staging_.resize(plane * static_cast<std::size_t>(nz_) * 2);
        // SERIAL on purpose: ses::parallel_for instantiated inside this
        // SHARED BASE module's inline member crashes a pool worker at
        // runtime (MSVC modules footgun -- the same call is fine in every
        // LEAF scene module; kin of the OpenMP-in-module-interface
        // miscompile that begat ses.parallel). The copy is ~2 MB: serial
        // costs ~1 ms and correctness beats it.
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const std::complex<double> z = psi_(i, j, 0);
                const std::size_t o =
                    2 * (static_cast<std::size_t>(j) *
                             static_cast<std::size_t>(nx) +
                         static_cast<std::size_t>(i));
                staging_[o] = static_cast<float>(z.real());
                staging_[o + 1] = static_cast<float>(z.imag());
            }
        }
        for (int k = 1; k < nz_; ++k) {
            std::copy(staging_.begin(),
                      staging_.begin() +
                          static_cast<std::ptrdiff_t>(plane * 2),
                      staging_.begin() +
                          static_cast<std::ptrdiff_t>(plane * 2) * k);
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
    int nz_;
    ses::Field3D psi_;
    std::vector<float> staging_;
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

private:
    ses::Mesh no_mesh_;
    std::vector<ses::Rgb> no_colors_;
};

}  // namespace ses_shell
