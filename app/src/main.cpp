// Humble Object shell -- the Qt boundary. Qt provides the window, input,
// the Vulkan device, and ONE fullscreen blit of the ses_vk renderer's scene
// image. Everything the demo IS lives in ses_shell::SimDirector
// (sim_director.hpp, Qt-free); the Viewport below translates events into
// director calls, assembles the renderer's FrameInput from director
// accessors, and formats the title.
//
// NO domain logic lives here (docs/ARCHITECTURE.md). The TDSE runs in core's
// WavepacketSimulation; matrices, colormaps, marching cubes, and ALL the
// volume-rendering math (ray_box, Beer-Lambert alpha, front-to-back
// compositing, phase LUT) live in core and are unit-tested -- the Vulkan-GLSL
// kernels under src/shaders/ are line-by-line transcriptions of those
// verified formulas.
//
// Controls: drag = orbit, wheel = zoom, space = pause, Tab = cloud/surface,
// 1 = real time, 2 = relax (imaginary time), 3 = relax to 2p, 4 = relax to
// 2s, 5 = excite an n=3 state (cascade demo), R = reset, M = measure
// position, E = measure energy, D = decay off/on, L = laser (off -> Z -> X
// -> off), F = flow particles, [ ] = thinner/denser cloud.

// ses_vk first: volk (inside) defines VK_NO_PROTOTYPES and must own the
// vulkan.h inclusion before any Qt header pulls its own Vulkan integration.
#include "vk_blobs.hpp"

#include "qt_blit.hpp"
#include "selftest_arcs.hpp"
#include "shell_ui.hpp"
#include "sim_director.hpp"
#include "vram_probe.hpp"

#include <QApplication>
#include <QKeyEvent>
#include <QMainWindow>
#include <QMessageBox>
#include <QMouseEvent>
#include <QRhiWidget>
#include <QString>
#include <QTimer>
#include <QVulkanInstance>
#include <QWheelEvent>

#include <rhi/qrhi_platform.h>  // QRhiVulkanNativeHandles (device adopt)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

[[noreturn]] void fatal_rhi_error(const char* stage, const QString& detail) {
    qCritical("%s: %s", stage, qPrintable(detail));
    QMessageBox::critical(nullptr, QStringLiteral("Graphics error"),
                          QStringLiteral("%1\n\n%2").arg(QLatin1String(stage), detail));
    std::exit(EXIT_FAILURE);
}

constexpr int kTickMs = 16;

class Viewport : public QRhiWidget {
public:
    explicit Viewport(QWidget* parent = nullptr) : QRhiWidget(parent) {
        setApi(QRhiWidget::Api::Vulkan);  // before the widget is first backed
        setFocusPolicy(Qt::StrongFocus);
        connect(&timer_, &QTimer::timeout, this, &Viewport::tick);
        timer_.start(kTickMs);
    }

protected:
    // The compute half of a frame runs BEFORE the widget's own QRhi frame is
    // recorded: the engine drives its own offscreen frames, which must not
    // interleave with the widget frame's recording. Once per paint, even
    // while paused (Key E works paused).
    void paintEvent(QPaintEvent* e) override {
        if (rhi_ready_) {
            if (!compute_init_done_) {
                init_compute();
                compute_init_done_ = true;
            }
            director_.run_frame();
            if (director_.take_title_dirty()) {
                refresh_title();
            }
            render_scene_offscreen();  // the whole scene, in ses_vk
        }
        QRhiWidget::paintEvent(e);
    }

    // The widget's QRhi (and thus the adopted VkDevice) is about to go away:
    // the core must tear down EVERYTHING it created on that device first.
    // Simulation state on the GPU dies with it -- the fatal-on-rhi-change
    // guard in initialize() keeps the policy honest (no silent migration).
    void releaseResources() override {
        blit_.release();
        vk_renderer_.destroy();
        vk_renderer_ready_ = false;
        director_.release_gpu();
        vk_ctx_.destroy();
    }

    // Build the RENDER-side Qt resources. Called by QRhiWidget once the
    // backing QRhi exists, and again on every resize -- the guard makes
    // re-entry a no-op (nothing here depends on the surface size; the
    // projection is per-frame). COMPUTE setup is deferred to init_compute():
    // the widget frame is ACTIVE during initialize(), and the engine drives
    // its own offscreen frames, which are illegal while a frame is recorded.
    void initialize(QRhiCommandBuffer* cb) override {
        if (rhi_ready_) {
            if (rhi() != rhi_) {
                fatal_rhi_error("QRhi changed",
                                QStringLiteral("The widget was moved to another window; "
                                               "GPU state cannot be migrated."));
            }
            return;
        }
        rhi_ = rhi();

        // The scene renders in ses_vk. Qt's whole render layer is the
        // BlitPresenter (one sampler + one fullscreen-blit pipeline, built
        // lazily once the scene image exists to seed the SRB).
        if (!blit_.init(rhi_)) {
            fatal_rhi_error("render resources",
                            QStringLiteral("blit sampler create failed"));
        }
        Q_UNUSED(cb);

        director_.mark_display_dirty();
        rhi_ready_ = true;
    }

    // Adopt the QRhiWidget's Vulkan device into the framework-free core:
    // Khronos handles cross the boundary, nothing else. Qt stays the device
    // provider + presentation layer; the physics runs in ses_vk via the
    // director.
    void init_compute() {
        const auto* h =
            static_cast<const QRhiVulkanNativeHandles*>(rhi_->nativeHandles());
        const bool adopted =
            h != nullptr && h->inst != nullptr &&
            vk_ctx_.adopt(h->inst->vkInstance(), h->physDev, h->dev,
                          h->gfxQueueFamilyIdx, h->gfxQueue) ==
                ses_vk::Boot::ok;
        // The scene renderer needs only the DEVICE, not the engine: the CPU
        // fallback path (engine init failure) still renders.
        vk_renderer_ready_ =
            adopted && vk_renderer_.initialize(vk_ctx_, director_.grid(),
                                               ses_shell::app_render_blobs());
        if (!vk_renderer_ready_) {
            fatal_rhi_error("render resources",
                            QStringLiteral("ses_vk renderer initialization "
                                           "failed (see stderr)"));
        }
        director_.init_compute(vk_ctx_, adopted,
                               ses_shell::query_free_vram_bytes(rhi()));
    }

    // The DRAW half of a frame, in ses_vk, outside any QRhi frame: assemble
    // the per-frame inputs (camera, view mode, brightness, staged uploads)
    // from the director, resize the offscreen target to the widget's pixel
    // size, and let SceneRenderer record the passes.
    void render_scene_offscreen() {
        if (!vk_renderer_ready_) {
            return;
        }
        const qreal dpr = devicePixelRatio();
        const std::uint32_t w =
            static_cast<std::uint32_t>(std::max<qreal>(1, width() * dpr));
        const std::uint32_t h =
            static_cast<std::uint32_t>(std::max<qreal>(1, height() * dpr));
        if (!vk_renderer_.resize(w, h)) {
            return;
        }

        ses_vk::SceneRenderer::FrameInput in;
        in.cloud = director_.cloud();
        in.azimuth = azimuth_;
        in.elevation = elevation_;
        in.distance = distance_;
        in.peak = director_.peak();
        in.absorbance = absorbance_;
        in.flash = director_.next_flash_intensity();
        // Probability-flow particles (Key F): drawn over the cloud, frozen
        // while paused so a still frame can still accumulate.
        in.flow = flow_on_ && in.cloud;
        in.flow_animate = !paused_;
        // Temporal accumulation: keep averaging only while NOTHING changed
        // -- camera, display params, view mode, flash, the psi volume
        // itself (any bridge write resets), and no animating particles.
        in.frame_index = static_cast<float>(frame_index_++ % 4096);
        const bool volume_written = director_.take_volume_written();
        const bool scene_static =
            !volume_written && azimuth_ == acc_prev_.azimuth &&
            elevation_ == acc_prev_.elevation &&
            distance_ == acc_prev_.distance && in.peak == acc_prev_.peak &&
            absorbance_ == acc_prev_.absorbance && in.flash == 0.0f &&
            acc_prev_.flash == 0.0f && in.cloud == acc_prev_.cloud;
        in.accumulate = scene_static && !(in.flow && in.flow_animate);
        // Occupancy + self-shadow rebuild when the field or the absorbance
        // dial (baked into the shadow transmittance) moved.
        in.volume_changed =
            volume_written || absorbance_ != acc_prev_.absorbance;
        acc_prev_ = {azimuth_, elevation_, distance_, in.peak, absorbance_,
                     in.flash, in.cloud};
        // The psi display volume: the engine's bridge image on the GPU path;
        // null lets the renderer fall back to its CPU-staged texture.
        in.psi_volume = director_.psi_volume_view();
        if (in.cloud) {
            if (director_.take_volume_dirty()) {
                // CPU staging only -- the bridge owns the volume on the GPU
                // path, and until compute init has been ATTEMPTED the 268 MB
                // fallback texture must not be allocated (it would be orphaned
                // one frame later and would deflate the VRAM-budget probe).
                if (director_.compute_attempted() && !director_.gpu_ok()) {
                    in.volume_staging = &director_.psi_staging();
                }
            }
        } else if (director_.take_mesh_dirty()) {
            in.mesh = &director_.mesh();
            in.mesh_colors = &director_.colors();
        }
        vk_renderer_.render(in);
    }

    // Qt's entire remaining draw lives in the BlitPresenter: one fullscreen
    // triangle sampling the ses_vk scene image.
    void render(QRhiCommandBuffer* cb) override {
        blit_.present(cb, renderTarget(), vk_renderer_.color_image(),
                      vk_renderer_.width(), vk_renderer_.height());
    }

    void mousePressEvent(QMouseEvent* e) override { last_pos_ = e->position(); }

    void mouseMoveEvent(QMouseEvent* e) override {
        const QPointF delta = e->position() - last_pos_;
        last_pos_ = e->position();
        azimuth_ -= 0.01 * delta.x();
        elevation_ += 0.01 * delta.y();
        elevation_ = std::clamp(elevation_, -1.5, 1.5);
        update();
    }

    void wheelEvent(QWheelEvent* e) override {
        distance_ *= std::pow(0.999, e->angleDelta().y());
        distance_ = std::clamp(distance_, 4.0, 300.0);  // out past the +-80 box
                                                         // (dynamic zfar follows)
        update();
    }

public:
    // Control entry points, shared by the keyboard, the toolbar buttons
    // (shell_ui.hpp), and the selftest arcs (selftest_arcs.hpp): thin
    // director delegations plus the Qt-side repaint/title refresh.
    void toggle_pause() {
        paused_ = !paused_;
        after_control();
    }
    void set_real_time() {
        director_.set_real_time();
        after_control();
    }
    void set_relaxing() {
        director_.set_relaxing();
        after_control();
    }
    void reset_simulation() {
        director_.reset_simulation();
        after_control();
    }
    void measure_now() {
        director_.measure_now();
        after_control();
    }
    void measure_energy_now() {
        director_.measure_energy_now();
        update();
    }
    void toggle_view_mode() {
        director_.toggle_view_mode();
        after_control();
    }
    void relax_to_excited() {
        director_.relax_to_excited();
        after_control();
    }
    void relax_to_2s() {
        director_.relax_to_2s();
        after_control();
    }
    void toggle_decay() {
        director_.toggle_decay();
        after_control();
    }
    void excite_n3() {
        director_.excite_n3();
        after_control();
    }
    void toggle_laser() {
        director_.toggle_laser();
        after_control();
    }
    void set_efield_e0(double e0) {
        director_.set_efield_e0(e0);
        after_control();
    }
    void set_bfield_b(double b) {
        director_.set_bfield_b(b);
        after_control();
    }
    void toggle_bfield_axis() {
        director_.toggle_bfield_axis();
        after_control();
    }
    int bfield_axis() const { return director_.bfield_axis(); }

    // Selftest / verification hooks: pure passthroughs (camera is the one
    // piece of state the shell owns itself).
    double channel_a(int from, int to) const { return director_.channel_a(from, to); }
    bool solving() const { return director_.solving(); }
    bool manifold_ready() const { return director_.manifold_ready(); }
    double state_energy(int idx) const { return director_.state_energy(idx); }
    long long photon_count() const { return director_.photon_count(); }
    int last_measured_index() const { return director_.last_measured_index(); }
    double mean_z() { return director_.mean_z(); }
    double peak_excited_population() const {
        return director_.peak_excited_population();
    }
    void debug_prepare_state(int idx) {
        director_.debug_prepare_state(idx);
        after_control();
    }
    double probe_population(int idx) { return director_.probe_population(idx); }

    // Verification hook (--dump-frame-near): place the camera at an explicit
    // distance -- inside the box (< ~80) exercises the volume proxy's
    // front-face culling, which only matters from an interior viewpoint.
    void debug_set_camera_distance(double d) {
        distance_ = std::clamp(d, 4.0, 300.0);
        update();
    }

protected:
    void after_control() {
        refresh_title();
        update();
    }

    void refresh_title() {
        window()->setWindowTitle(QString::fromStdString(director_.title_text()));
    }

    void keyPressEvent(QKeyEvent* e) override {
        switch (e->key()) {
            case Qt::Key_Space:
                toggle_pause();
                break;
            case Qt::Key_1:
                set_real_time();
                break;
            case Qt::Key_2:
                set_relaxing();
                break;
            case Qt::Key_3:
                relax_to_excited();
                break;
            case Qt::Key_4:
                relax_to_2s();
                break;
            case Qt::Key_5:
                excite_n3();
                break;
            case Qt::Key_R:
                reset_simulation();
                break;
            case Qt::Key_M:
                measure_now();
                break;
            case Qt::Key_E:
                measure_energy_now();
                break;
            case Qt::Key_D:
                toggle_decay();
                break;
            case Qt::Key_F:
                // Probability-current flow particles (Bohmian tracers):
                // still for real eigenstates, swirling under drive/B-field.
                flow_on_ = !flow_on_;
                update();
                break;
            case Qt::Key_L:
                toggle_laser();
                break;
            case Qt::Key_Tab:
                toggle_view_mode();
                break;
            case Qt::Key_BracketLeft:
                absorbance_ = std::max(0.1, absorbance_ / 1.3);
                after_control();
                break;
            case Qt::Key_BracketRight:
                absorbance_ = std::min(50.0, absorbance_ * 1.3);
                after_control();
                break;
            default:
                QRhiWidget::keyPressEvent(e);
                return;
        }
    }

private:
    void tick() {
        if (paused_) {
            return;
        }
        director_.tick();
        if (director_.take_title_dirty()) {
            refresh_title();
        }
        update();
    }

    // Qt-side GPU plumbing: the adopted device context, the framework-free
    // scene renderer, and the one-blit presenter. The context is declared
    // FIRST: members destroy in reverse order, and everything below allocates
    // from its VMA allocator, so the context must outlive them all.
    ses_vk::DeviceContext vk_ctx_;  // adopts the QRhiWidget's device
    ses_vk::SceneRenderer vk_renderer_;  // the whole scene, framework-free
    bool vk_renderer_ready_ = false;

    // The whole demo, Qt-free (sim_director.hpp): CPU truth session, ses_vk
    // engine, atom model, and the demo state machine. Declared after the
    // device context (the engine's buffers die before the allocator).
    ses_shell::SimDirector director_;
    ses_shell::BlitPresenter blit_;
    QRhi* rhi_ = nullptr;             // the widget's QRhi, cached in initialize()
    bool rhi_ready_ = false;          // render resources built
    bool compute_init_done_ = false;  // director init_compute ran (first paint)

    // Temporal-accumulation bookkeeping.
    struct AccumPrev {
        double azimuth = 1e9, elevation = 0, distance = 0, peak = 0,
               absorbance = 0;
        float flash = 0.0f;
        bool cloud = true;
    } acc_prev_;
    long long frame_index_ = 0;
    bool flow_on_ = false;  // Key F: probability-current flow particles

    QTimer timer_;
    bool paused_ = false;
    double absorbance_ = 0.68;  // lightened from 1.5: the default cloud was
                                // too opaque

    double azimuth_ = 0.6;
    double elevation_ = 0.4;
    double distance_ = 150.0;  // default frames ~+-62 Bohr (45 deg fovy): the
                               // n<=6 manifold body, incl. the ~60 Bohr 6s/6p
                               // (wheel in toward 4 for a close-up of small ones)
    QPointF last_pos_;
};

}  // namespace

int main(int argc, char** argv) {
    // GUI-subsystem stderr through a redirect is a fully buffered pipe; a
    // crash then eats every diagnostic. Unbuffered keeps them honest.
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    // QRhi on the Vulkan backend (Viewport::setApi). No MSAA: the volume ray
    // marcher is a full-screen fragment pass where 4x multisampling only
    // multiplies its cost (it smooths nothing but the cube edges) -- at 256^3
    // that budget belongs to the physics. QRhiWidget's sampleCount default (1)
    // already matches.
    QApplication app(argc, argv);

    QMainWindow window;
    window.setWindowTitle(QStringLiteral("Electron wavepacket near a hydrogen nucleus"));
    auto* viewport = new Viewport();
    window.setCentralWidget(viewport);

    // Discoverable controls (shell_ui.hpp): toolbar buttons mirroring the
    // hotkeys + the two field sliders, all Qt::NoFocus so the keys stay live.
    ses_shell::build_control_bar(window, viewport);

    window.resize(1024, 768);
    window.show();
    // StrongFocus only ACCEPTS focus; grab it so the 1/2/R/M/space keys work
    // immediately without a click.
    viewport->setFocus();

    // Verification + selftest arcs (--dump-frame*, --selftest-*): registered
    // from selftest_arcs.hpp so main() stays a shell.
    ses_shell::register_verification_arcs(app, viewport);

    return app.exec();
}
