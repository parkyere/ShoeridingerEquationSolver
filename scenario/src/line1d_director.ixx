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
export module ses.scenario.line1d_director;
export import ses.scenario;
export import ses.field;
export import ses.grid;
export import ses.observables;
export import ses.phasor;
export import ses.potential;
export import ses.propagator;


// Shared base for the six textbook 1D scenes (HO ladder, tunneling,
// double well, Poschl-Teller, Morse, Bloch). Physics is pure CPU double
// -- even the 64k-point split-operator step is cheap, so no engine is
// involved and gpu_ok() stays false by design. Display goes through the overlay polyline
// seam: the wavefunction as the white phasor curve (radius = r_scale |psi|^2,
// twist = arg psi -- ON THE CURVE phase is geometry, never color), its
// |psi|^2 shadow band on the z = 0 plane (phase as per-vertex hue there,
// the volume view's wheel -- the plane is where color-coded phase lives),
// and the potential as the red profile in the z = 0 plane. The 3D
// machinery (volume, mesh, flow) is disabled wholesale: cloud() false, no
// staging, no marker.
// volk.h textually first: VK_* macros never cross module boundaries.


export namespace ses_shell {

class Line1DDirector : public ScenarioDirector {
public:
    // ---- lifecycle ----
    const ses::Grid3D& grid() const override { return grid3d_; }
    void init_compute(ses_vk::DeviceContext& /*ctx*/, bool /*device_ok*/,
                      std::int64_t /*free_vram_bytes*/) override {
        compute_attempted_ = true;  // CPU-only scene: nothing to set up
    }
    void release_gpu() override {}
    bool compute_attempted() const override { return compute_attempted_; }
    bool gpu_ok() const override { return false; }

    // ---- per-frame / per-tick ----
    void run_frame() override {
        if (pending_steps_ > 0) {
            const int n = pending_steps_;
            pending_steps_ = 0;
            step_batch(n);
            sim_time_ += n * dt_;
            display_changed_ = true;
            psi_dirty_ = true;
            after_batch();
        }
        // Rebuild the phasor curve and its plane shadow only when psi
        // actually moved (steps, key mutations, resets): the 64k-point
        // color pass is real per-frame money, and a paused scene recomputes
        // byte-identical arrays otherwise.
        if (std::exchange(psi_dirty_, false)) {
            rebuild_psi_display();
        }
        if (title_due_) {
            title_due_ = false;
            title_dirty_ = true;
        }
    }
    void tick() override {
        // One tick's supply per rendered frame (the BaseDirector pacing
        // rule): catch-up ticks drop instead of bundling.
        const int per_tick = steps_per_tick() * time_scale_;
        pending_steps_ = std::min(pending_steps_ + per_tick, per_tick);
        if (++ticks_ % 10 == 0) {
            title_due_ = true;
        }
    }

    // ---- generic controls ----
    void do_set_real_time() override {}  // no-op: NVI base already resets time_scale to 1
    void reset_simulation() override {
        psi_ = initial_;
        sim_time_ = 0.0;
        pending_steps_ = 0;
        display_changed_ = true;
        psi_dirty_ = true;
        title_dirty_ = true;
        after_reset();
    }
    void measure_now() override {}
    void toggle_view_mode() override {}  // single (overlay) view
    bool handle_key(char /*key*/) override { return false; }

    // ---- shell gating ----
    bool solving() const override { return false; }
    bool scene_ready() const override { return true; }

    void set_time_scale(int scale) override {
        time_scale_ = std::clamp(scale, 1, 16);
    }
    int time_scale() const override { return time_scale_; }
    double sim_time() const override { return sim_time_; }
    double sim_dt() const override { return dt_; }

    // ---- display accessors ----
    bool cloud() const override { return false; }
    double peak() const override { return 1.0; }
    VkImageView psi_volume_view() override { return VK_NULL_HANDLE; }
    float next_flash_intensity() override { return 0.0f; }
    bool take_volume_written() override {
        return std::exchange(display_changed_, false);
    }
    bool take_volume_dirty() override { return false; }
    bool take_mesh_dirty() override { return false; }
    // Scene code calls this after mutating psi in place (ladder rungs):
    // the psi-derived curves must follow.
    void mark_display_dirty() override {
        display_changed_ = true;
        psi_dirty_ = true;
    }
    bool take_title_dirty() override { return std::exchange(title_dirty_, false); }
    const std::vector<float>& psi_staging() const override { return no_staging_; }
    const ses::Mesh& mesh() const override { return no_mesh_; }
    const std::vector<ses::Rgb>& colors() const override { return no_colors_; }
    std::string title_text() override {
        return strf("%s  |  t = %.1f au (dt %.3g, %d pts)%s", scene_name(),
                    sim_time_, dt_, grid1d_.n, title_suffix().c_str());
    }

    int marker_count() const override { return 0; }  // a line has no nucleus

    int overlay_curve_count() const override { return 4; }
    OverlayCurve overlay_curve(int i) const override {
        if (i == 0) {  // faint xy (z = 0) reference sheet, drawn first
            return {plane_quad_.data(), 4, 0.45f, 0.55f, 0.75f, 0.07f, true};
        }
        if (i == 1) {
            // |psi|^2 projected onto the plane as the phasor tube's shadow
            // band, phase as per-vertex hue (the volume view's wheel) --
            // constant rgba here is ignored, the color array wins.
            return {dens_band_.data(), 2 * grid1d_.n, 1.0f, 1.0f,
                    1.0f,              1.0f,          true, dens_cols_.data()};
        }
        if (i == 2) {  // the potential profile: warm red, slightly translucent
            return {pot_curve_.data(), grid1d_.n, 1.0f, 0.30f, 0.25f, 0.9f};
        }
        // The wavefunction: white phasor curve, on top.
        return {psi_curve_.data(), grid1d_.n, 1.0f, 1.0f, 1.0f, 1.0f};
    }

protected:
    // e_scale/y_clamp: potential-curve display mapping (Ha -> Bohr height).
    Line1DDirector(const ses::Grid1D& g, std::vector<double> potential,
                   double dt, double r_scale, double e_scale, double y_clamp)
        : grid1d_(g),
          grid3d_{g, g, g},  // cubic display frame around the line
          potential_(std::move(potential)),
          dt_(dt),
          r_scale_(r_scale),
          e_scale_(e_scale),
          y_clamp_(y_clamp),
          psi_(g),
          initial_(g) {
        prop_ = std::make_unique<ses::SplitOperator1D>(grid1d_, potential_, dt_);
        pot_curve_ = ses::potential_curve(grid1d_, potential_, e_scale_, y_clamp_);
        // Faint z = 0 (xy) reference sheet: makes the phasor twist legible
        // (in-plane vs out-of-plane), and face-on -- the Z snap view -- it
        // frames the textbook 2D plot. Spans the box in x, a quarter box
        // in y, as one triangle-strip quad.
        const float hh = static_cast<float>(0.25 * grid1d_.xmax);
        const float x0 = static_cast<float>(grid1d_.xmin);
        const float x1 = static_cast<float>(grid1d_.xmax);
        plane_quad_ = {x0, -hh, 0.0f, x1, -hh, 0.0f,
                       x0, hh,  0.0f, x1, hh,  0.0f};
    }

    // Scenario-specific hooks.
    virtual const char* scene_name() const = 0;
    virtual std::string title_suffix() { return ""; }
    virtual int steps_per_tick() const { return 1; }
    virtual void after_batch() {}
    virtual void after_reset() {}

    // The initial state; also the reset target. Builds the first phasor
    // curve immediately so overlay_curve() is valid before any run_frame.
    void set_state(ses::Field1D f) {
        initial_ = f;
        psi_ = std::move(f);
        rebuild_psi_display();
        display_changed_ = true;
    }
    // Boundary absorber (tunneling): psi *= mask each step.
    void set_mask(std::vector<double> mask) { mask_ = std::move(mask); }

    // Swap the potential on the SAME grid/dt: rebuilds the propagator
    // tables and the red profile. psi is deliberately untouched -- a
    // sudden quench is legitimate physics (the state persists, the well
    // changes under it).
    void set_potential(std::vector<double> v) {
        potential_ = std::move(v);
        prop_ = std::make_unique<ses::SplitOperator1D>(grid1d_, potential_, dt_);
        pot_curve_ = ses::potential_curve(grid1d_, potential_, e_scale_, y_clamp_);
        display_changed_ = true;
        title_dirty_ = true;
    }
    // Retarget reset without touching the live psi (quench bookkeeping).
    void set_reset_target(ses::Field1D f) { initial_ = std::move(f); }
    // Replace the live psi without touching the reset target.
    void replace_state(ses::Field1D f) {
        psi_ = std::move(f);
        rebuild_psi_display();
        display_changed_ = true;
        title_dirty_ = true;
    }

    // The white curve, its plane-shadow band, and the band's phase hues
    // move together with psi.
    void rebuild_psi_display() {
        psi_curve_ = ses::phasor_curve(psi_, r_scale_);
        dens_band_ = ses::density_band(psi_, r_scale_);
        dens_cols_ = ses::phase_band_colors(psi_, kBandAlpha);
    }

    // Overridable: the Bloch scene substitutes its tilted propagator.
    virtual void step_batch(int n) {
        for (int s = 0; s < n; ++s) {
            prop_->step(psi_);
            if (!mask_.empty()) {
                for (int i = 0; i < psi_.size(); ++i) {
                    psi_[i] *= mask_[static_cast<std::size_t>(i)];
                }
            }
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

    ses::Grid1D grid1d_;
    ses::Grid3D grid3d_;
    std::vector<double> potential_;
    double dt_;
    double r_scale_;
    double e_scale_;
    double y_clamp_;
    std::unique_ptr<ses::SplitOperator1D> prop_;
    ses::Field1D psi_;
    ses::Field1D initial_;
    std::vector<double> mask_;
    double sim_time_ = 0.0;
    int pending_steps_ = 0;
    int time_scale_ = 1;
    std::uint64_t ticks_ = 0;
    bool title_due_ = false;
    bool title_dirty_ = true;
    bool display_changed_ = true;
    bool psi_dirty_ = true;  // psi moved since the last curve rebuild
    bool compute_attempted_ = false;

    // Loud enough to read hue, quiet enough that the white curve and the
    // red potential stay the foreground.
    static constexpr float kBandAlpha = 0.35f;

    std::vector<float> psi_curve_;
    std::vector<float> pot_curve_;
    std::vector<float> plane_quad_;
    std::vector<float> dens_band_;
    std::vector<float> dens_cols_;

private:
    std::vector<float> no_staging_;
    ses::Mesh no_mesh_;
    std::vector<ses::Rgb> no_colors_;
};

}  // namespace ses_shell
