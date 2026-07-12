// Humble Object shell -- the Qt boundary. Qt provides the window, input, the
// Vulkan device, and ONE fullscreen blit of the ses_vk renderer's scene
// image; everything the demo IS lives behind ses_shell::ScenarioDirector
// (Qt-free; --scene= picks the implementation). NO domain logic lives here
// (docs/ARCHITECTURE.md).
//
// Controls: drag = orbit, wheel = zoom, space = pause, Tab = cloud/surface,
// 1 = real time, 2 = relax (imaginary time), 3 = relax to 2p, 4 = relax to
// 2s, 5 = excite an n=3 state (cascade demo), R = reset, M = measure
// position, E = measure energy, D = decay off/on, L = laser (off -> Z -> X
// -> off), F = flow particles, [ ] = thinner/denser cloud.

// ses_vk first: volk (inside) defines VK_NO_PROTOTYPES and must own the
// vulkan.h inclusion before any Qt header pulls its own Vulkan integration.
#include "vk_blobs.hpp"

#include "harmonic_director.hpp"
#include "hydrogen_director.hpp"
#include "qt_blit.hpp"
#include "selftest_arcs.hpp"
#include "shell_ui.hpp"
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
#include <memory>

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
    explicit Viewport(std::unique_ptr<ses_shell::ScenarioDirector> director,
                      QWidget* parent = nullptr)
        : QRhiWidget(parent), director_(std::move(director)) {
        hydrogen_ = dynamic_cast<ses_shell::HydrogenDirector*>(director_.get());
        distance_ = director_->default_camera_distance();
        setApi(QRhiWidget::Api::Vulkan);  // before the widget is first backed
        setFocusPolicy(Qt::StrongFocus);
        connect(&timer_, &QTimer::timeout, this, &Viewport::tick);
        timer_.start(kTickMs);
    }

protected:
    // The compute half runs BEFORE the widget's QRhi frame is recorded (the
    // engine's offscreen frames must not interleave with it). Once per
    // paint, even while paused (Key E works paused).
    void paintEvent(QPaintEvent* e) override {
        if (rhi_ready_) {
            if (!compute_init_done_) {
                init_compute();
                compute_init_done_ = true;
            }
            director_->run_frame();
            if (director_->take_title_dirty()) {
                refresh_title();
            }
            render_scene_offscreen();  // the whole scene, in ses_vk
        }
        QRhiWidget::paintEvent(e);
    }

    // The adopted VkDevice is about to go away: tear down everything the
    // core created on it first. GPU simulation state dies with it (the
    // fatal-on-rhi-change guard in initialize() forbids silent migration).
    void releaseResources() override {
        blit_.release();
        vk_renderer_.destroy();
        vk_renderer_ready_ = false;
        director_->release_gpu();
        vk_ctx_.destroy();
    }

    // Build the RENDER-side Qt resources; re-entry (every resize) is a no-op.
    // COMPUTE setup is deferred to init_compute(): the widget frame is ACTIVE
    // during initialize(), and engine offscreen frames are illegal mid-frame.
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

        // Qt's whole render layer is the BlitPresenter (one sampler + one
        // fullscreen-blit pipeline, built lazily once the scene image exists).
        if (!blit_.init(rhi_)) {
            fatal_rhi_error("render resources",
                            QStringLiteral("blit sampler create failed"));
        }
        Q_UNUSED(cb);

        director_->mark_display_dirty();
        rhi_ready_ = true;
    }

    // Adopt the QRhiWidget's Vulkan device into the framework-free core:
    // Khronos handles cross the boundary, nothing else.
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
            adopted && vk_renderer_.initialize(vk_ctx_, director_->grid(),
                                               ses_shell::app_render_blobs());
        if (!vk_renderer_ready_) {
            fatal_rhi_error("render resources",
                            QStringLiteral("ses_vk renderer initialization "
                                           "failed (see stderr)"));
        }
        director_->init_compute(
            vk_ctx_, adopted,
            ses_shell::query_free_vram_bytes(
                adopted ? vk_ctx_.phys_dev : VK_NULL_HANDLE));
    }

    // The DRAW half, in ses_vk, outside any QRhi frame: assemble FrameInput
    // from the director, resize the offscreen target, record the passes.
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
        in.cloud = director_->cloud();
        in.azimuth = azimuth_;
        in.elevation = elevation_;
        in.distance = distance_;
        in.peak = director_->peak();
        in.absorbance = absorbance_;
        in.flash = director_->next_flash_intensity();
        // Probability-flow particles (Key F): drawn over the cloud, frozen
        // while paused so a still frame can still accumulate.
        in.flow = flow_on_ && in.cloud;
        in.flow_animate = !paused_;
        // Temporal accumulation: keep averaging only while NOTHING changed
        // (camera, display params, flash, psi volume, animating particles).
        in.frame_index = static_cast<float>(frame_index_++ % 4096);
        const bool volume_written = director_->take_volume_written();
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
        in.psi_volume = director_->psi_volume_view();
        if (in.cloud) {
            if (director_->take_volume_dirty()) {
                // CPU staging only: until compute init has been ATTEMPTED the
                // 268 MB fallback texture must not be allocated (it would be
                // orphaned and would deflate the VRAM-budget probe).
                if (director_->compute_attempted() && !director_->gpu_ok()) {
                    in.volume_staging = &director_->psi_staging();
                }
            }
        } else if (director_->take_mesh_dirty()) {
            in.mesh = &director_->mesh();
            in.mesh_colors = &director_->colors();
        }
        vk_renderer_.render(in);
    }

    // Qt's entire remaining draw: one fullscreen triangle sampling the scene.
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
        director_->set_real_time();
        after_control();
    }
    void set_relaxing() {
        if (hydrogen_) hydrogen_->set_relaxing();
        after_control();
    }
    void reset_simulation() {
        director_->reset_simulation();
        after_control();
    }
    void measure_now() {
        director_->measure_now();
        after_control();
    }
    void measure_energy_now() {
        if (hydrogen_) hydrogen_->measure_energy_now();
        update();
    }
    void toggle_view_mode() {
        director_->toggle_view_mode();
        after_control();
    }
    void relax_to_excited() {
        if (hydrogen_) hydrogen_->relax_to_excited();
        after_control();
    }
    void relax_to_2s() {
        if (hydrogen_) hydrogen_->relax_to_2s();
        after_control();
    }
    void toggle_decay() {
        if (hydrogen_) hydrogen_->toggle_decay();
        after_control();
    }
    void excite_n3() {
        if (hydrogen_) hydrogen_->excite_n3();
        after_control();
    }
    void toggle_laser() {
        if (hydrogen_) hydrogen_->toggle_laser();
        after_control();
    }
    void set_efield_e0(double e0) {
        if (hydrogen_) hydrogen_->set_efield_e0(e0);
        after_control();
    }
    void set_bfield_b(double b) {
        if (hydrogen_) hydrogen_->set_bfield_b(b);
        after_control();
    }
    void toggle_bfield_axis() {
        if (hydrogen_) hydrogen_->toggle_bfield_axis();
        after_control();
    }
    int bfield_axis() const { return hydrogen_ ? hydrogen_->bfield_axis() : 2; }

    // Selftest / verification hooks: pure passthroughs (camera is the one
    // piece of state the shell owns itself).
    double channel_a(int from, int to) const {
        return hydrogen_ ? hydrogen_->channel_a(from, to) : 0.0;
    }
    bool solving() const { return director_->solving(); }
    bool manifold_ready() const { return director_->scene_ready(); }
    double state_energy(int idx) const {
        return hydrogen_ ? hydrogen_->state_energy(idx) : 0.0;
    }
    long long photon_count() const {
        return hydrogen_ ? hydrogen_->photon_count() : 0;
    }
    int last_measured_index() const {
        return hydrogen_ ? hydrogen_->last_measured_index() : -2;
    }
    double mean_z() { return hydrogen_ ? hydrogen_->mean_z() : 0.0; }
    double peak_excited_population() const {
        return hydrogen_ ? hydrogen_->peak_excited_population() : 0.0;
    }
    void debug_prepare_state(int idx) {
        if (hydrogen_) hydrogen_->debug_prepare_state(idx);
        after_control();
    }
    double probe_population(int idx) {
        return hydrogen_ ? hydrogen_->probe_population(idx) : 0.0;
    }

    // Verification hook (--dump-frame-near): an interior camera (< ~80)
    // exercises the volume proxy's front-face culling.
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
        window()->setWindowTitle(QString::fromStdString(director_->title_text()));
    }

    // Generic keys live here; everything else is offered to the scenario as
    // a plain ASCII key (hydrogen: 2/3/4/5/D/E/L).
    void keyPressEvent(QKeyEvent* e) override {
        switch (e->key()) {
            case Qt::Key_Space:
                toggle_pause();
                return;
            case Qt::Key_1:
                set_real_time();
                return;
            case Qt::Key_R:
                reset_simulation();
                return;
            case Qt::Key_M:
                measure_now();
                return;
            case Qt::Key_F:
                // Probability-current flow particles (Bohmian tracers).
                flow_on_ = !flow_on_;
                update();
                return;
            case Qt::Key_Tab:
                toggle_view_mode();
                return;
            case Qt::Key_BracketLeft:
                absorbance_ = std::max(0.1, absorbance_ / 1.3);
                after_control();
                return;
            case Qt::Key_BracketRight:
                absorbance_ = std::min(50.0, absorbance_ * 1.3);
                after_control();
                return;
            default:
                break;
        }
        const int k = e->key();
        const char ch =
            (k >= Qt::Key_0 && k <= Qt::Key_9)
                ? static_cast<char>('0' + (k - Qt::Key_0))
                : (k >= Qt::Key_A && k <= Qt::Key_Z)
                      ? static_cast<char>('A' + (k - Qt::Key_A))
                      : '\0';
        if (ch != '\0' && director_->handle_key(ch)) {
            after_control();
            return;
        }
        QRhiWidget::keyPressEvent(e);
    }

public:
    // Toolbar entry point: feed a scenario key as if typed.
    void press(char ch) {
        if (director_->handle_key(ch)) {
            after_control();
        }
    }

private:
    void tick() {
        if (paused_) {
            return;
        }
        director_->tick();
        if (director_->take_title_dirty()) {
            refresh_title();
        }
        update();
    }

    // The context is declared FIRST: members destroy in reverse order, and
    // everything below allocates from its VMA allocator, so the context must
    // outlive them all.
    ses_vk::DeviceContext vk_ctx_;  // adopts the QRhiWidget's device
    ses_vk::SceneRenderer vk_renderer_;  // the whole scene, framework-free
    bool vk_renderer_ready_ = false;

    // The scenario (scenario.hpp, chosen in main by --scene=): everything
    // the demo IS, Qt-free. Declared after the device context (the engine's
    // buffers die before the allocator). hydrogen_ is the narrow view the
    // hydrogen-only wrappers (toolbar + selftest arcs) delegate through;
    // null for other scenes, where those wrappers no-op.
    std::unique_ptr<ses_shell::ScenarioDirector> director_;
    ses_shell::HydrogenDirector* hydrogen_ = nullptr;
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
    double absorbance_ = 0.68;  // lightened from 1.5 (default was too opaque)

    double azimuth_ = 0.6;
    double elevation_ = 0.4;
    double distance_ = 150.0;  // frames ~+-62 Bohr at 45 deg fovy: the n<=6
                               // manifold body
    QPointF last_pos_;
};

}  // namespace

int main(int argc, char** argv) {
    // GUI-subsystem stderr through a redirect is a fully buffered pipe; a
    // crash then eats every diagnostic. Unbuffered keeps them honest.
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    // No MSAA: the ray marcher is a full-screen fragment pass where 4x only
    // smooths cube edges; QRhiWidget's sampleCount default (1) already fits.
    QApplication app(argc, argv);

    // Scenario selection (--scene=hydrogen|...); the selftest arcs all drive
    // the hydrogen scene, which is also the default.
    QString scene = QStringLiteral("hydrogen");
    for (const QString& a : app.arguments()) {
        if (a.startsWith(QStringLiteral("--scene="))) {
            scene = a.mid(8);
        }
    }
    std::unique_ptr<ses_shell::ScenarioDirector> director;
    if (scene == QStringLiteral("harmonic")) {
        director = std::make_unique<ses_shell::HarmonicDirector>();
    } else {
        if (scene != QStringLiteral("hydrogen")) {
            std::fprintf(stderr, "scene: unknown '%s' -- using hydrogen\n",
                         qPrintable(scene));
            scene = QStringLiteral("hydrogen");
        }
        director = std::make_unique<ses_shell::HydrogenDirector>();
    }

    QMainWindow window;
    window.setWindowTitle(QStringLiteral("Electron wavepacket near a hydrogen nucleus"));
    auto* viewport = new Viewport(std::move(director));
    window.setCentralWidget(viewport);

    // Discoverable controls (shell_ui.hpp): toolbar buttons mirroring the
    // hotkeys, all Qt::NoFocus so the keys stay live.
    if (scene == QStringLiteral("hydrogen")) {
        ses_shell::build_control_bar(window, viewport);
    } else {
        ses_shell::build_generic_bar(
            window, viewport,
            {qMakePair(QStringLiteral("Relax to ground (2)"), '2')});
    }

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
