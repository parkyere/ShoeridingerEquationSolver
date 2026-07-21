// Humble Object shell: owns window+input (SDL3), device+swapchain+main loop
// (ses.vk), and the ImGui panel. No domain logic -- scenes live behind
// ses_shell::ScenarioDirector, --scene= picks one (docs/ARCHITECTURE.md).

// Std headers first, in full: a FIRST textual include after the ses.* module
// imports' GMFs would C2572 (MSVC C++20-modules ordering).
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>

// volk before SDL/ImGui: it owns the vulkan.h include (VK_NO_PROTOTYPES).
// Single VMA_IMPLEMENTATION TU lives in solver/src/vma_impl.cpp (not here).
#include <volk.h>  // VK_* macros: modules cannot export them

#include <blit_frag_spv.h>
#include <blit_vert_spv.h>

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <SDL3/SDL_main.h>
import ses.scenario.anderson1d_director;
import ses.scenario.billiard2d_director;
import ses.scenario.bouncer1d_director;
import ses.scenario.carpet1d_director;
import ses.scenario.qpc2d_director;
import ses.scenario.spin_director;
import ses.scenario.spins_director;
import ses.scenario.bloch1d_director;
import ses.scenario.corral2d_director;
import ses.scenario.doubleslit2d_director;
import ses.scenario.landau2d_director;
import ses.scenario.qdot2d_director;
import ses.scenario.rutherford3d_director;
import ses.scenario.tunneling_director;

import app.scheduler;
import ses.scenario.doublewell1d_director;
import ses.scenario.harmonic_director;
import ses.scenario.harmonic1d_director;
import ses.scenario.hydrogen_director;
import ses.scenario.molecule_director;
import ses.scenario.morse1d_director;
import ses.scenario.ptwell1d_director;
import ses.scenario.tunneling1d_director;
import ses.scenario.selftest_arcs;
import ses.camera;
import ses.vk.render_blobs;
import app.imgui_ui;
import ses.vk.present;
import ses.vk.vram_probe;

namespace {

[[noreturn]] void fatal_shell_error(const char* stage, const char* detail) {
    std::fprintf(stderr, "%s: %s\n", stage, detail);
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, stage, detail, nullptr);
    std::exit(EXIT_FAILURE);
}

constexpr std::uint64_t kTickMs = 16;

// Panel-combo order; make_scene_director switches on this index.
constexpr const char* kSceneNames[] = {
    "hydrogen", "harmonic", "tunnel",  "harmonic1d",   "tunnel1d",
    "doublewell1d", "ptwell1d", "morse1d", "h2plus",   "benzene",
    "doubleslit2d", "landau2d", "bloch1d", "corral2d", "qdot2d",
    "billiard2d", "anderson1d", "carpet1d", "qpc2d", "bouncer1d",
    "spin", "spins", "rutherford3d"};
constexpr int kSceneCount = 23;
// Combo display labels; SAME order and length as kSceneNames (static_assert
// below turns any drift into a compile error).
constexpr const char* kSceneLabels[] = {
    "Hydrogen atom", "Harmonic trap", "Tunneling barrier",
    "1D harmonic oscillator", "1D tunneling barrier", "1D double well",
    "1D reflectionless well", "1D Morse well", "H2+ molecular ion",
    "Stripped benzene (1e)", "2D double slit + AB", "2D Landau / cyclotron",
    "1D crystal lattice (Bloch)", "2D quantum corral", "2D quantum dot",
    "2D quantum billiard", "1D Anderson localization", "1D quantum carpet",
    "2D quantum point contact", "1D quantum bouncer", "Electron spin (Bloch)",
    "16 interacting spins", "Rutherford scattering"};
static_assert(sizeof(kSceneNames) / sizeof(kSceneNames[0]) == kSceneCount,
              "kSceneNames length != kSceneCount");
static_assert(sizeof(kSceneLabels) / sizeof(kSceneLabels[0]) == kSceneCount,
              "kSceneLabels length != kSceneCount (combo out of sync)");
std::unique_ptr<ses_shell::ScenarioDirector> make_scene_director(int idx) {
    switch (idx) {
        case 1: return std::make_unique<ses_shell::HarmonicDirector>();
        case 2: return std::make_unique<ses_shell::TunnelingDirector>();
        case 3: return std::make_unique<ses_shell::Harmonic1DDirector>();
        case 4: return std::make_unique<ses_shell::Tunneling1DDirector>();
        case 5: return std::make_unique<ses_shell::DoubleWell1DDirector>();
        case 6: return std::make_unique<ses_shell::PtWell1DDirector>();
        case 7: return std::make_unique<ses_shell::Morse1DDirector>();
        case 8: return std::make_unique<ses_shell::H2PlusDirector>();
        case 9: return std::make_unique<ses_shell::BenzeneDirector>();
        case 10: return std::make_unique<ses_shell::DoubleSlit2DDirector>();
        case 11: return std::make_unique<ses_shell::Landau2DDirector>();
        case 12: return std::make_unique<ses_shell::Bloch1DDirector>();
        case 13: return std::make_unique<ses_shell::Corral2DDirector>();
        case 14: return std::make_unique<ses_shell::Qdot2DDirector>();
        case 15: return std::make_unique<ses_shell::Billiard2DDirector>();
        case 16: return std::make_unique<ses_shell::Anderson1DDirector>();
        case 17: return std::make_unique<ses_shell::Carpet1DDirector>();
        case 18: return std::make_unique<ses_shell::Qpc2DDirector>();
        case 19: return std::make_unique<ses_shell::Bouncer1DDirector>();
        case 20: return std::make_unique<ses_shell::SpinDirector>();
        case 21: return std::make_unique<ses_shell::SpinsDirector>();
        case 22: return std::make_unique<ses_shell::Rutherford3DDirector>();
        default: return std::make_unique<ses_shell::HydrogenDirector>();
    }
}

// --scene name -> combo index, or -1 if unknown.
int scene_index_of(const std::string& name) {
    for (int i = 0; i < kSceneCount; ++i) {
        if (name == kSceneNames[i]) {
            return i;
        }
    }
    return -1;
}

// Arcs that force a NON-hydrogen scene (else a mismatched --scene could leave
// the arc dereferencing a null capability). Arcs absent here drive hydrogen.
struct ArcScene {
    const char* arc;
    const char* scene;
};
constexpr ArcScene kArcScenes[] = {
    {"selftest-tunnel", "tunnel"},
    {"selftest-trapdecay", "harmonic"},
    {"selftest-ladder1d", "harmonic1d"},
    {"selftest-tunnel1d", "tunnel1d"},
    {"selftest-dw1d", "doublewell1d"},
    {"selftest-pt1d", "ptwell1d"},
    {"selftest-morse1d", "morse1d"},
    {"selftest-doubleslit2d", "doubleslit2d"},
    {"selftest-landau", "landau2d"},
    {"selftest-bloch", "bloch1d"},
    {"selftest-corral", "corral2d"},
    {"selftest-qdot", "qdot2d"},
    {"selftest-billiard", "billiard2d"},
    {"selftest-anderson", "anderson1d"},
    {"selftest-cat", "harmonic1d"},
    {"selftest-carpet", "carpet1d"},
    {"selftest-qpc2d", "qpc2d"},
    {"selftest-bouncer", "bouncer1d"},
    {"selftest-spin", "spin"},
    {"selftest-spins", "spins"},
    {"selftest-h2p", "h2plus"},
    {"selftest-h2p-orbitals", "h2plus"},
    {"selftest-benzene", "benzene"},
    {"selftest-rutherford", "rutherford3d"},
};

class Shell {
public:
    Shell(int scene_index, std::vector<std::string> args)
        : director_(make_scene_director(scene_index)),
          scene_index_(scene_index),
          args_(std::move(args)) {
        distance_ = director_->default_camera_distance();
        azimuth_ = director_->default_camera_azimuth();
        elevation_ = director_->default_camera_elevation();
        // Verification arcs run HEADLESS (no window/surface/swapchain/ImGui):
        // presenting would tie the loop to vsync + Windows' occlusion throttle,
        // which strangles the sim rate and false-fails wall-clock verdicts.
        // --dump-frame still renders, but OFFSCREEN.
        for (const std::string& a : args_) {
            if (a.rfind("--selftest-", 0) == 0 ||
                a.rfind("--dump-frame", 0) == 0) {
                headless_ = true;
            }
            if (a.rfind("--dump-frame", 0) == 0) {
                needs_render_ = true;
            }
        }
        flow_on_ = has_arg("--flow");
        if (has_arg("--face-z")) {
            snap_camera_z();
        }
    }

    void init() {
        // SES_VK_VALIDATION=1: validation layer for the WINDOWED render path,
        // the only way to check dynamic-rendering/present/sync2 barriers that
        // sesolver_vkcheck (compute-only) can't reach.
        const char* venv = std::getenv("SES_VK_VALIDATION");
        const bool want_validation = (venv != nullptr && venv[0] == '1');
        if (headless_) {
            if (!SDL_Init(SDL_INIT_EVENTS)) {  // SDL_GetTicks + event drain
                fatal_shell_error("SDL init", SDL_GetError());
            }
            if (vk_ctx_.create(want_validation) != ses_vk::Boot::ok) {
                fatal_shell_error("Vulkan device",
                                  "no Vulkan runtime on this machine");
            }
            std::fprintf(stderr, "vk: device %s (headless)%s\n",
                         vk_ctx_.device_name,
                         vk_ctx_.validation_active ? " [validation ON]" : "");
            if (needs_render_ &&
                !vk_renderer_.initialize(vk_ctx_, director_->grid(),
                                         ses_vk::render_blobs())) {
                fatal_shell_error("render resources",
                                  "ses_vk renderer initialization failed");
            }
            refresh_status();
            return;
        }

        if (!SDL_Init(SDL_INIT_VIDEO)) {
            fatal_shell_error("SDL init", SDL_GetError());
        }
        // 2048 wide: keeps the ImGui panel clear of the centered cloud.
        window_ = SDL_CreateWindow(
            "Electron wavepacket near a hydrogen nucleus", 2048, 768,
            SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE |
                SDL_WINDOW_HIGH_PIXEL_DENSITY);
        if (window_ == nullptr) {
            fatal_shell_error("SDL window", SDL_GetError());
        }

        Uint32 n_ext = 0;
        const char* const* sdl_exts = SDL_Vulkan_GetInstanceExtensions(&n_ext);
        if (sdl_exts == nullptr) {
            fatal_shell_error("Vulkan surface extensions", SDL_GetError());
        }
        const std::vector<const char*> exts{sdl_exts, sdl_exts + n_ext};
        if (vk_ctx_.create_instance(want_validation, exts) != ses_vk::Boot::ok) {
            fatal_shell_error("Vulkan instance",
                              "no Vulkan runtime on this machine");
        }
        if (!SDL_Vulkan_CreateSurface(window_, vk_ctx_.instance, nullptr,
                                      &surface_)) {
            fatal_shell_error("Vulkan surface", SDL_GetError());
        }
        if (vk_ctx_.create_device(surface_) != ses_vk::Boot::ok) {
            fatal_shell_error("Vulkan device",
                              "no compute+present queue family");
        }
        std::fprintf(stderr, "vk: device %s%s\n", vk_ctx_.device_name,
                     vk_ctx_.validation_active ? " [validation ON]" : "");

        if (!vk_renderer_.initialize(vk_ctx_, director_->grid(),
                                     ses_vk::render_blobs())) {
            fatal_shell_error("render resources",
                              "ses_vk renderer initialization failed");
        }
        if (!presenter_.init(vk_ctx_, surface_, k_blit_vert_spv,
                             k_blit_vert_spv_size, k_blit_frag_spv,
                             k_blit_frag_spv_size)) {
            fatal_shell_error("render resources",
                              "swapchain presenter initialization failed");
        }
        init_imgui();
        refresh_status();
        // Compute init is DEFERRED into the loop: VkFFT plan compile + radial
        // solve block for seconds, and the window must present a frame first,
        // not sit black behind them.
    }

    void shutdown() {
        vkDeviceWaitIdle(vk_ctx_.device);
        if (!headless_) {
            ImGui_ImplVulkan_Shutdown();
            ImGui_ImplSDL3_Shutdown();
            ImGui::DestroyContext();
            presenter_.release();
        }
        vk_renderer_.destroy();
        director_->release_gpu();
        if (surface_ != VK_NULL_HANDLE) {
            SDL_Vulkan_DestroySurface(vk_ctx_.instance, surface_, nullptr);
            surface_ = VK_NULL_HANDLE;
        }
        vk_ctx_.destroy();
        if (window_ != nullptr) {
            SDL_DestroyWindow(window_);
            window_ = nullptr;
        }
        SDL_Quit();
    }

    int run() {
        last_tick_ = SDL_GetTicks();
        while (!exit_requested_) {
            pump_events();
            apply_pending_scene();
            if (!headless_) {
                // Acquire EARLY so the vsync wait overlaps the sim batch +
                // render below instead of stalling in present().
                presenter_.acquire();
            }
            const std::uint64_t now = SDL_GetTicks();
            // Fixed 16 ms ticks; coalesce after a stall instead of spiraling.
            if (now - last_tick_ > 8 * kTickMs) {
                last_tick_ = now - kTickMs;
            }
            while (now - last_tick_ >= kTickMs) {
                last_tick_ += kTickMs;
                if (!paused_) {
                    director_->tick();
                }
            }
            sched_.poll(now);
            if (exit_requested_) {
                break;
            }

            // Deferred compute init: headless goes immediately; windowed waits
            // for one presented frame so the seconds-long init isn't behind a
            // black window.
            if (!compute_init_done_ && (headless_ || presented_once_)) {
                director_->init_compute(
                    vk_ctx_, true,
                    ses_shell::query_free_vram_bytes(vk_ctx_.phys_dev));
                compute_init_done_ = true;
                refresh_status();
            }

            if (compute_init_done_) {
                // Compute half runs before any render (offscreen engine frames
                // must not interleave a render), once per frame even while
                // paused (Key E works paused). Physics arcs skip rendering.
                director_->run_frame();
                if (director_->take_title_dirty()) {
                    refresh_status();
                }
                if (!headless_ || needs_render_) {
                    render_scene_offscreen();
                }
            }

            if (headless_) {
                SDL_Delay(1);  // don't busy-spin between fixed-cadence ticks
                continue;
            }

            // Achieved sim rate over a ~1 s rolling window (panel readout).
            if (now - perf_last_ms_ >= 1000) {
                const double st = director_->sim_time();
                perf_sim_rate_ = (st - perf_last_sim_t_) * 1000.0 /
                                 static_cast<double>(now - perf_last_ms_);
                perf_last_sim_t_ = st;
                perf_last_ms_ = now;
            }

            // UI + present: the scene blits in the presenter's pass, ImGui
            // rides the same pass after it.
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplSDL3_NewFrame();
            ImGui::NewFrame();
            if (auto* hy = director_->hydrogen()) {
                app::draw_hydrogen_panel(*this, ui_, *hy);
            } else if (auto* ld = director_->ladder1d()) {
                app::draw_ladder1d_panel(*this, ui_, *ld);
            } else if (auto* dwp = director_->doublewell()) {
                app::draw_doublewell_panel(*this, ui_, *dwp);
            } else if (director_->morse() != nullptr) {
                app::draw_generic_panel(*this, ui_,
                                        {{"Jump up (U)", 'U'},
                                         {"Jump down (D)", 'D'},
                                         {"Pair beat (S)", 'S'},
                                         {"Ground (2)", '2'}});
            } else if (director_->reflect() != nullptr) {
                app::draw_generic_panel(*this, ui_,
                                        {{"Swap well (W)", 'W'}});
            } else if (auto* mol = director_->molecule()) {
                if (scene_index_ == 8) {
                    app::draw_h2plus_panel(*this, ui_, *mol);
                } else {
                    app::draw_benzene_panel(*this, ui_, *mol);
                }
            } else if (auto* slit = director_->slit()) {
                app::draw_doubleslit_panel(*this, ui_, *slit);
            } else if (auto* lnd = director_->landau()) {
                app::draw_landau_panel(*this, ui_, *lnd);
            } else if (auto* blc = director_->bloch()) {
                app::draw_bloch_panel(*this, ui_, *blc);
            } else if (auto* crl = director_->corral()) {
                app::draw_corral_panel(*this, ui_, *crl);
            } else if (auto* qdt = director_->qdot()) {
                app::draw_qdot_panel(*this, ui_, *qdt);
            } else if (auto* bil = director_->billiard()) {
                app::draw_billiard_panel(*this, ui_, *bil);
            } else if (auto* and1 = director_->anderson()) {
                app::draw_anderson_panel(*this, ui_, *and1);
            } else if (director_->carpet() != nullptr) {
                app::draw_generic_panel(*this, ui_,
                                        {{"Refire (2)", '2'}});
            } else if (auto* qpc = director_->qpc()) {
                app::draw_qpc_panel(*this, ui_, *qpc);
            } else if (director_->bouncer() != nullptr) {
                app::draw_generic_panel(*this, ui_,
                                        {{"Airy ground (2)", '2'},
                                         {"Drop (F)", 'F'}});
            } else if (auto* spn = director_->spin()) {
                app::draw_spin_panel(*this, ui_, *spn);
            } else if (auto* sns = director_->spins()) {
                app::draw_spins_panel(*this, ui_, *sns);
            } else if (auto* rf = director_->rutherford()) {
                app::draw_rutherford_panel(*this, ui_, *rf);
            } else if (director_->tunnel() != nullptr) {
                app::draw_generic_panel(*this, ui_, {});
            } else {
                app::draw_generic_panel(
                    *this, ui_,
                    {{"Relax to ground (2)", '2'},
                     {"Excite N (5)", '5'},
                     {"Decay (D)", 'D'},
                     {"Measure E (E)", 'E'}});
            }
            // Refresh here too: a panel control changed while PAUSED (no tick,
            // no title-dirty) would otherwise read stale forever.
            refresh_status();
            ImGui::Render();
            ImDrawData* dd = ImGui::GetDrawData();
            const bool presented = presenter_.present(
                vk_renderer_.color_view(), [dd](VkCommandBuffer cb) {
                    ImGui_ImplVulkan_RenderDrawData(dd, cb);
                });
            if (presented) {
                presented_once_ = true;
            } else {
                SDL_Delay(10);  // minimized / swapchain rebuilding
            }
        }
        return exit_code_;
    }

    // Generic control entry points; scenario-specific ones live behind
    // director_->hydrogen()/tunnel(), reached by the panel and arcs.
    void toggle_pause() { paused_ = !paused_; }
    void set_real_time() {
        director_->set_real_time();
        refresh_status();
    }
    void reset_simulation() {
        director_->reset_simulation();
        refresh_status();
    }
    void measure_now() {
        director_->measure_now();
        refresh_status();
    }
    void toggle_view_mode() {
        director_->toggle_view_mode();
        refresh_status();
    }
    // Key Z: face +z (azimuth 0, elevation 0) so the 1D scenes' xy sheet
    // spans the screen plane.
    void snap_camera_z() {
        azimuth_ = 0.0;
        elevation_ = 0.0;
    }
    bool cloud_view() const { return director_ && director_->cloud(); }
    // Selftest hook: z-normal slice sheet through the nucleus so lobe signs
    // read clearly.
    void enable_cross_section_demo() {
        ui_.slice_on = true;
        ui_.slice_axis = 2;
        ui_.slice_map = 0;     // density
    }
    void set_time_scale(int scale) {
        director_->set_time_scale(scale);
        refresh_status();
    }
    // Director truth for the panel slider: programmatic changes (set_real_time
    // reset, clamping) must show, not the last drag.
    int time_scale() const { return director_->time_scale(); }
    // Panel readout: achieved au/s and the x1 baseline it's measured against.
    double sim_rate() const { return perf_sim_rate_; }
    double baseline_sim_rate() const {
        return (1000.0 / kTickMs) * director_->sim_dt() *
               director_->steps_per_tick_x1();
    }

    // ImGui panel entry point: feed a scenario key as if typed.
    void press(char ch) {
        if (director_->handle_key(ch)) {
            refresh_status();
        }
    }

    int scene_index() const { return scene_index_; }
    static int scene_count() { return kSceneCount; }
    static const char* scene_label(int i) { return kSceneLabels[i]; }
    void request_scene(int idx) {
        if (idx >= 0 && idx < kSceneCount) {
            pending_scene_ = idx;
        }
    }

    // Scenario + capability seams (single accessors); the arcs call e.g.
    // hy()->channel_a().
    ses_shell::ScenarioDirector& director() { return *director_; }
    ses_shell::HydrogenApi* hy() { return director_->hydrogen(); }
    ses_shell::TunnelApi* tn() { return director_->tunnel(); }
    ses_shell::Ladder1dApi* ln() { return director_->ladder1d(); }
    ses_shell::DoubleWellApi* dw() { return director_->doublewell(); }
    ses_shell::ReflectApi* rf() { return director_->reflect(); }
    ses_shell::MorseApi* mo() { return director_->morse(); }
    ses_shell::MoleculeApi* ml() { return director_->molecule(); }
    ses_shell::SlitApi* sl() { return director_->slit(); }
    ses_shell::LandauApi* la() { return director_->landau(); }
    ses_shell::BlochApi* bl() { return director_->bloch(); }
    ses_shell::CorralApi* cr() { return director_->corral(); }
    ses_shell::QdotApi* qd() { return director_->qdot(); }
    ses_shell::BilliardApi* bd() { return director_->billiard(); }
    ses_shell::AndersonApi* an() { return director_->anderson(); }
    ses_shell::CarpetApi* cp() { return director_->carpet(); }
    ses_shell::QpcApi* qp() { return director_->qpc(); }
    ses_shell::BouncerApi* bo() { return director_->bouncer(); }
    ses_shell::SpinApi* sp() { return director_->spin(); }
    ses_shell::SpinsApi* sn() { return director_->spins(); }
    bool solving() const { return director_->solving(); }
    bool manifold_ready() const { return director_->scene_ready(); }
    void debug_set_camera_distance(double d) {
        distance_ = std::clamp(d, 4.0, 300.0);
    }

    app::Scheduler& sched() { return sched_; }
    void request_exit(int code) {
        exit_code_ = code;
        exit_requested_ = true;
    }
    bool has_arg(const char* name) const {
        for (const std::string& a : args_) {
            if (a == name) {
                return true;
            }
        }
        return false;
    }
    bool dump_frame_bmp(const char* path) {
        return ses_shell::dump_scene_bmp(vk_ctx_, vk_renderer_.color_image(),
                                         vk_renderer_.width(),
                                         vk_renderer_.height(), path);
    }
    std::uint32_t frame_width() const { return vk_renderer_.width(); }
    std::uint32_t frame_height() const { return vk_renderer_.height(); }
    const std::string& status_text() const { return status_text_; }

private:
    // Deferred scene switch: director owns live GPU work, so wait for compute
    // init, idle the device before teardown, then re-run the SAME deferred
    // compute-init path (window stays live through the new scene's solve).
    void apply_pending_scene() {
        if (pending_scene_ < 0 || !compute_init_done_) {
            return;
        }
        const int idx = pending_scene_;
        pending_scene_ = -1;
        if (idx == scene_index_) {
            return;
        }
        std::fprintf(stderr, "scene: switching to %s\n", kSceneNames[idx]);
        vkDeviceWaitIdle(vk_ctx_.device);
        director_->release_gpu();
        director_ = make_scene_director(idx);
        scene_index_ = idx;
        if (!headless_ || needs_render_) {
            // Grid-shaped resources must match the new scene: full renderer
            // rebuild (destroy() resets its memos -- reinit-safety pinned by
            // the --dump-frame-switch arc).
            vk_renderer_.destroy();
            if (!vk_renderer_.initialize(vk_ctx_, director_->grid(),
                                         ses_vk::render_blobs())) {
                fatal_shell_error("render resources",
                                  "renderer re-init after scene switch failed");
            }
        }
        ui_ = app::UiState{};  // sliders must not carry stale field values
        distance_ = director_->default_camera_distance();
        azimuth_ = director_->default_camera_azimuth();
        elevation_ = director_->default_camera_elevation();
        acc_prev_ = {};
        paused_ = false;
        compute_init_done_ = false;
        refresh_status();
    }

    void init_imgui() {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::GetIO().IniFilename = nullptr;  // no imgui.ini litter
        ImGui::StyleColorsDark();
        ImGui_ImplSDL3_InitForVulkan(window_);

        ImGui_ImplVulkan_InitInfo info{};
        info.ApiVersion = VK_API_VERSION_1_4;
        info.Instance = vk_ctx_.instance;
        info.PhysicalDevice = vk_ctx_.phys_dev;
        info.Device = vk_ctx_.device;
        info.QueueFamily = vk_ctx_.queue_family;
        info.Queue = vk_ctx_.queue;
        // ImGui 1.92 backend allocates SAMPLED_IMAGE + SAMPLER (not COMBINED):
        // DescriptorPoolSize > 0 with DescriptorPool null lets it own a
        // correctly typed pool (freed in ImGui_ImplVulkan_Shutdown).
        info.DescriptorPoolSize = 16;
        // Dynamic rendering (no VkRenderPass): the UI pipeline declares the
        // swapchain color format. ImGui 1.92 deep-copies the format array in
        // Init, so a local is safe.
        info.UseDynamicRendering = true;
        info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        const VkFormat ui_color_fmt = presenter_.color_format();
        info.PipelineInfoMain.PipelineRenderingCreateInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
        info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats =
            &ui_color_fmt;
        info.MinImageCount = presenter_.min_image_count();
        info.ImageCount = std::max(presenter_.image_count(),
                                   presenter_.min_image_count());
        if (!ImGui_ImplVulkan_Init(&info)) {
            fatal_shell_error("render resources", "ImGui Vulkan init failed");
        }
    }

    void pump_events() {
        if (headless_) {  // no window, no ImGui context: just drain the queue
            SDL_Event e;
            while (SDL_PollEvent(&e)) {
                if (e.type == SDL_EVENT_QUIT) {
                    request_exit(exit_code_);
                }
            }
            return;
        }
        ImGuiIO& io = ImGui::GetIO();
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            ImGui_ImplSDL3_ProcessEvent(&e);
            switch (e.type) {
                case SDL_EVENT_QUIT:
                    request_exit(exit_code_);
                    break;
                case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                    presenter_.request_resize();
                    break;
                case SDL_EVENT_KEY_DOWN:
                    if (!io.WantCaptureKeyboard && !e.key.repeat) {
                        handle_key(e.key.key);
                    }
                    break;
                case SDL_EVENT_MOUSE_BUTTON_DOWN:
                    // RIGHT-drag on a 2D-HO surface point: grab the cloud
                    // (pulling UP gathers the packet).
                    if (!io.WantCaptureMouse &&
                        e.button.button == SDL_BUTTON_RIGHT) {
                        if (auto* qd = director_->qdot()) {
                            double gx = 0.0;
                            double gy = 0.0;
                            if (pick_stage(e.button.x, e.button.y, &gx,
                                           &gy)) {
                                qd->begin_grab(gx, gy);
                                grab_pixels_ = 0.0;
                            }
                        }
                    }
                    break;
                case SDL_EVENT_MOUSE_BUTTON_UP:
                    if (e.button.button == SDL_BUTTON_RIGHT) {
                        if (auto* qd = director_->qdot()) {
                            qd->end_grab();
                        }
                    }
                    break;
                case SDL_EVENT_MOUSE_MOTION:
                    if (auto* qd = director_->qdot();
                        qd != nullptr && qd->grabbing() &&
                        (e.motion.state & SDL_BUTTON_RMASK) != 0) {
                        grab_pixels_ -= e.motion.yrel;  // up = gather
                        qd->update_grab(
                            std::clamp(grab_pixels_ / 250.0, 0.0, 1.0));
                        break;
                    }
                    if (!io.WantCaptureMouse &&
                        (e.motion.state & SDL_BUTTON_LMASK) != 0) {
                        azimuth_ -= 0.01 * e.motion.xrel;
                        elevation_ += 0.01 * e.motion.yrel;
                        elevation_ = std::clamp(elevation_, -1.5, 1.5);
                    }
                    break;
                case SDL_EVENT_MOUSE_WHEEL:
                    if (!io.WantCaptureMouse) {
                        // Per-notch zoom: SDL gives ~1 unit/notch; 120 sets
                        // ~11% per notch.
                        distance_ *=
                            std::pow(0.999, 120.0 * e.wheel.y);
                        distance_ = std::clamp(distance_, 4.0, 300.0);
                    }
                    break;
                default:
                    break;
            }
        }
    }

    // Window point -> z=0 stage plane via the orbit camera (ses.camera
    // unproject_to_z0; CONTRACT: pick_test).
    bool pick_stage(float mx, float my, double* out_x, double* out_y) {
        int w = 0;
        int h = 0;
        SDL_GetWindowSize(window_, &w, &h);
        if (w <= 0 || h <= 0) {
            return false;
        }
        const double kPi = 3.14159265358979323846;
        const double ndc_x = 2.0 * mx / w - 1.0;
        const double ndc_y = 1.0 - 2.0 * my / h;
        return ses::unproject_to_z0(azimuth_, elevation_, distance_,
                                    45.0 * kPi / 180.0,
                                    static_cast<double>(w) / h, ndc_x,
                                    ndc_y, out_x, out_y);
    }

    // Generic keys handled here; anything else is offered to the scenario as
    // a plain ASCII key.
    void handle_key(SDL_Keycode key) {
        switch (key) {
            case SDLK_SPACE:
                toggle_pause();
                return;
            case SDLK_1:
                set_real_time();
                return;
            case SDLK_R:
                reset_simulation();
                return;
            case SDLK_M:
                measure_now();
                return;
            case SDLK_F:
                // Scene hotkey first; an unhandled F toggles the flow tracers
                // (Bohmian current).
                if (director_->handle_key('F')) {
                    return;
                }
                flow_on_ = !flow_on_;
                return;
            case SDLK_TAB:
                toggle_view_mode();
                return;
            case SDLK_Z:
                snap_camera_z();
                return;
            case SDLK_LEFTBRACKET:
                absorbance_ = std::max(0.1, absorbance_ / 1.3);
                return;
            case SDLK_RIGHTBRACKET:
                absorbance_ = std::min(50.0, absorbance_ * 1.3);
                return;
            default:
                break;
        }
        char ch = '\0';
        if (key >= SDLK_0 && key <= SDLK_9) {
            ch = static_cast<char>('0' + (key - SDLK_0));
        } else if (key >= SDLK_A && key <= SDLK_Z) {
            ch = static_cast<char>('A' + (key - SDLK_A));
        }
        if (ch != '\0' && director_->handle_key(ch)) {
            refresh_status();
        }
    }

    void refresh_status() { status_text_ = director_->title_text(); }

    // The DRAW half (ses_vk): render() is synchronous -- the presenter
    // samples afterwards.
    void render_scene_offscreen() {
        int pw = 1024;  // headless --dump-frame: no window, fixed extent
        int ph = 768;
        if (window_ != nullptr) {
            SDL_GetWindowSizeInPixels(window_, &pw, &ph);
        }
        const std::uint32_t w = static_cast<std::uint32_t>(std::max(1, pw));
        const std::uint32_t h = static_cast<std::uint32_t>(std::max(1, ph));
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
        // Flow particles (Key F): over the cloud, frozen while paused so a
        // still frame can still accumulate.
        in.flow = flow_on_ && in.cloud;
        in.flow_animate = !paused_;
        in.clip_on = ui_.clip_on;
        in.clip_axis = ui_.clip_axis;
        in.clip_sign = ui_.clip_sign;
        in.clip_offset = ui_.clip_offset;
        in.slice_on = ui_.slice_on;
        in.slice_axis = ui_.slice_axis;
        in.slice_offset = ui_.slice_offset;
        in.slice_map = ui_.slice_map;
        // Packed scalar: changes whenever any plane control moves, so the
        // accumulation early-out treats a plane edit as "not static".
        const double plane_tag =
            (ui_.clip_on ? 1.0 : 0.0) + 2.0 * ui_.clip_axis +
            8.0 * ui_.clip_sign + 100.0 * ui_.clip_offset +
            (ui_.slice_on ? 1000.0 : 0.0) + 4000.0 * ui_.slice_axis +
            200000.0 * ui_.slice_offset + 30000000.0 * ui_.slice_map;
        // Temporal accumulation: average only while NOTHING changed (camera,
        // display params, flash, psi volume, animating particles).
        in.frame_index = static_cast<float>(frame_index_++ % 4096);
        const bool volume_written = director_->take_volume_written();
        const bool scene_static =
            !volume_written && azimuth_ == acc_prev_.azimuth &&
            elevation_ == acc_prev_.elevation &&
            distance_ == acc_prev_.distance && in.peak == acc_prev_.peak &&
            absorbance_ == acc_prev_.absorbance && in.flash == 0.0f &&
            acc_prev_.flash == 0.0f && in.cloud == acc_prev_.cloud &&
            plane_tag == acc_prev_.plane_tag;
        in.accumulate = scene_static && !(in.flow && in.flow_animate);
        // Occupancy + self-shadow rebuild when the field or the absorbance
        // dial (baked into the shadow transmittance) moved.
        in.volume_changed =
            volume_written || absorbance_ != acc_prev_.absorbance;
        acc_prev_ = {azimuth_, elevation_, distance_, in.peak, absorbance_,
                     in.flash, in.cloud, plane_tag};
        // 1D-scene overlay polylines (phasor curve + potential); 3D scenes
        // report 0 and skip.
        const int nov = std::min(director_->overlay_curve_count(),
                                 ses_vk::SceneRenderer::kMaxOverlayCurves);
        for (int c = 0; c < nov; ++c) {
            const ses_shell::OverlayCurve oc = director_->overlay_curve(c);
            in.overlay[c] = {oc.xyz, oc.count, oc.r,    oc.g,
                             oc.b,   oc.a,     oc.fill, oc.rgba};
        }
        in.overlay_count = nov;
        const int nmk = std::min(director_->marker_count(),
                                 ses_vk::SceneRenderer::kMaxMarkers);
        for (int m = 0; m < nmk; ++m) {
            const ses_shell::SceneMarker mk = director_->marker(m);
            in.markers[m] = {mk.x, mk.y, mk.z, mk.radius, mk.r, mk.g, mk.b};
        }
        in.marker_count = nmk;
        double barrier_lo = 0.0;
        double barrier_hi = 0.0;
        in.barrier_on = director_->barrier_slab(barrier_lo, barrier_hi);
        in.barrier_lo = static_cast<float>(barrier_lo);
        in.barrier_hi = static_cast<float>(barrier_hi);
        // Psi display volume: the engine's GPU bridge image; null falls back
        // to the renderer's CPU-staged texture.
        in.psi_volume = director_->psi_volume_view();
        in.flow_velocity = director_->flow_velocity_view();
        if (in.cloud) {
            if (director_->take_volume_dirty()) {
                // Until compute init has been ATTEMPTED, don't allocate the
                // 268 MB fallback texture -- it would be orphaned and deflate
                // the VRAM-budget probe.
                if (director_->compute_attempted() && !director_->gpu_ok()) {
                    in.volume_staging = &director_->psi_staging();
                }
            }
        } else if (director_->surface_vbuf() != VK_NULL_HANDLE) {
            // GPU-extracted isosurface: drawn indirect, no host mesh upload.
            in.gpu_mesh_vbuf = director_->surface_vbuf();
            in.gpu_mesh_indirect = director_->surface_indirect();
        } else if (director_->take_mesh_dirty()) {
            in.mesh = &director_->mesh();
            in.mesh_colors = &director_->colors();
        }
        vk_renderer_.render(in);
    }

    // Context declared FIRST: members destroy in reverse order and all
    // allocate from its VMA allocator, so it must outlive them.
    ses_vk::DeviceContext vk_ctx_;
    ses_vk::SceneRenderer vk_renderer_;
    ses_shell::SwapchainPresenter presenter_;

    std::unique_ptr<ses_shell::ScenarioDirector> director_;
    int scene_index_ = 0;
    int pending_scene_ = -1; // panel-requested switch, applied at frame top

    SDL_Window* window_ = nullptr;
    double grab_pixels_ = 0.0;  // grab drag height (px, up = positive)
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;

    std::vector<std::string> args_;
    app::Scheduler sched_;
    app::UiState ui_;
    std::string status_text_;
    double perf_sim_rate_ = 0.0;     // achieved au/s, ~1 s rolling window
    double perf_last_sim_t_ = 0.0;
    std::uint64_t perf_last_ms_ = 0;
    bool headless_ = false;      // --selftest-*/--dump-frame: no window/ImGui
    bool needs_render_ = false;  // --dump-frame: offscreen renders required
    bool compute_init_done_ = false;
    bool presented_once_ = false;     // gates the deferred compute init
    bool exit_requested_ = false;
    int exit_code_ = 0;
    std::uint64_t last_tick_ = 0;

    struct AccumPrev {
        double azimuth = 1e9, elevation = 0, distance = 0, peak = 0,
               absorbance = 0;
        float flash = 0.0f;
        bool cloud = true;
        double plane_tag = 0.0;  // cross-section controls fingerprint
    } acc_prev_;
    long long frame_index_ = 0;
    bool flow_on_ = false;  // Key F: probability-current flow particles
    bool paused_ = false;
    double absorbance_ = 0.68;  // Beer-Lambert opacity of the |psi|^2 fog

    double azimuth_ = 0.6;
    double elevation_ = 0.4;
    double distance_ = 150.0;  // frames ~+-62 Bohr at 45 deg fovy (n<=6 body)
};

}  // namespace

int main(int argc, char* argv[]) {
    // GUI-subsystem stderr through a redirect is fully buffered; a crash then
    // eats every diagnostic. Unbuffered keeps them honest.
    std::setvbuf(stderr, nullptr, _IONBF, 0);

    // CLI parsing via Boost.Program_options (infrastructure is not physics --
    // standing rule). Every arc flag is registered: --help lists them and a
    // typo is a hard error, not a silently ignored dead flag.
    namespace po = boost::program_options;
    po::options_description desc("sesolver options");
    auto add = desc.add_options();
    add("help", "print this list and exit");
    add("scene", po::value<std::string>()->default_value("hydrogen"),
        "boot scene (hydrogen, harmonic, tunnel, harmonic1d, tunnel1d, "
        "doublewell1d, ptwell1d, morse1d, h2plus, benzene, doubleslit2d, "
        "landau2d, bloch1d, corral2d, qdot2d, billiard2d, anderson1d, "
        "carpet1d, qpc2d, bouncer1d, spin, spins)");
    add("face-z", "boot straight into the z-facing (textbook) view");
    add("flow", "start with the probability-flow streaklines on");
    for (const char* flag :
         {"dump-frame", "dump-frame-late", "dump-frame-mid", "dump-frame-near",
          "dump-frame-slice", "dump-frame-surface", "dump-frame-switch"}) {
        add(flag, "render verification arc (offscreen frame dump)");
    }
    for (const char* flag :
         {"selftest-anderson", "selftest-benzene", "selftest-billiard",
          "selftest-bouncer",
          "selftest-bloch", "selftest-carpet", "selftest-cascade",
          "selftest-cat",
          "selftest-corral", "selftest-decay", "selftest-doubleslit2d",
          "selftest-dw1d", "selftest-efield", "selftest-energy",
          "selftest-h2p", "selftest-h2p-orbitals", "selftest-kepler",
          "selftest-ladder1d", "selftest-landau",
          "selftest-magnetic", "selftest-manifold", "selftest-morse1d",
          "selftest-partial", "selftest-pt1d", "selftest-qdot",
          "selftest-qpc2d",
          "selftest-rabi", "selftest-rutherford", "selftest-scene",
          "selftest-spin", "selftest-spins", "selftest-trapdecay",
          "selftest-tunnel", "selftest-tunnel1d"}) {
        add(flag, "physics verification arc (headless)");
    }
    po::variables_map vm;
    try {
        // No abbreviation guessing: a shortened --selftest-cor would parse
        // here but never match the arcs' exact has_arg() lookups, booting a
        // normal window instead of the arc.
        po::store(po::command_line_parser(argc, argv)
                      .options(desc)
                      .style(po::command_line_style::default_style &
                             ~po::command_line_style::allow_guessing)
                      .run(),
                  vm);
        po::notify(vm);
    } catch (const po::error& e) {
        std::fprintf(stderr, "arguments: %s\n", e.what());
        std::ostringstream help;
        help << desc;
        std::fprintf(stderr, "%s", help.str().c_str());
        // Release builds run in the GUI subsystem (stderr detached), so a flag
        // typo must not exit SILENTLY.
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "arguments", e.what(),
                                 nullptr);
        return 2;
    }
    if (vm.count("help") > 0) {
        std::ostringstream help;
        help << desc;
        std::fprintf(stderr, "%s", help.str().c_str());
        return 0;
    }

    // The shell and the arcs keep their flag-string seam (has_arg).
    std::vector<std::string> args{argv, argv + argc};

    // Scene selection: --scene, overridden by any arc with its own scene
    // (kArcScenes); every OTHER selftest arc drives hydrogen. Plain
    // --dump-frame keeps the requested scene.
    std::string scene = vm["scene"].as<std::string>();
    bool arc_forced = false;
    for (const ArcScene& as : kArcScenes) {
        if (vm.count(as.arc) > 0) {
            scene = as.scene;
            arc_forced = true;
            break;
        }
    }
    if (!arc_forced) {
        for (const std::string& a : args) {
            if (a.rfind("--selftest-", 0) == 0) {
                scene = "hydrogen";
                break;
            }
        }
    }
    int scene_index = scene_index_of(scene);
    if (scene_index < 0) {
        std::fprintf(stderr, "scene: unknown '%s' -- using hydrogen\n",
                     scene.c_str());
        scene_index = 0;
    }

    Shell shell{scene_index, std::move(args)};
    shell.init();

    // Verification arcs registered from ses.scenario.selftest_arcs so main()
    // stays a shell.
    ses_shell::register_verification_arcs(&shell);

    const int code = shell.run();
    shell.shutdown();
    return code;
}
