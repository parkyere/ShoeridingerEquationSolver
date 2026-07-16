// Humble Object shell -- the SDL3 boundary. SDL provides the window, input,
// and the Vulkan surface; the shell OWNS the device (DeviceContext::create),
// the swapchain (ses.vk.present), and the main loop; Dear ImGui draws the
// control panel + status readout inside the presenter's pass. Everything the
// demo IS lives behind ses_shell::ScenarioDirector (--scene= picks the
// implementation). NO domain logic lives here (docs/ARCHITECTURE.md).
//
// Controls: drag = orbit, wheel = zoom, space = pause, Tab = cloud/surface,
// 1 = real time, 2 = relax (imaginary time), 3 = relax to 2p, 4 = relax to
// 2s, 5 = excite an n=3 state (cascade demo), R = reset, M = measure
// position, E = measure energy, D = decay off/on, L = laser (off -> Z -> X
// -> off), F = flow particles, [ ] = thinner/denser cloud.

// Std first, in full: the imported ses.* modules' GMFs reach these std
// headers -- a later FIRST textual include would C2572, so this TU textually
// claims the whole set up front (later ones are guard no-ops).
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
#include <string>
#include <thread>
#include <utility>
#include <vector>

// volk before SDL/ImGui: it defines VK_NO_PROTOTYPES and must own the
// vulkan.h inclusion before SDL/ImGui pull their own Vulkan declarations.
// The single VMA implementation TU is solver/src/vma_impl.cpp (ses_solver).
#include <volk.h>  // VK_* macros: modules cannot export them

#include <blit_frag_spv.h>
#include <blit_vert_spv.h>

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <SDL3/SDL_main.h>
import ses.scenario.tunneling_director;

import app.scheduler;
import ses.scenario.harmonic_director;
import ses.scenario.hydrogen_director;
import ses.scenario.selftest_arcs;
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

// The shell: window + input + device + presentation + the main loop. The one
// wrapper surface shared by the keyboard, the ImGui panel (app.imgui_ui),
// and the selftest arcs (ses.scenario.selftest_arcs).
class Shell {
public:
    Shell(std::unique_ptr<ses_shell::ScenarioDirector> director,
          std::vector<std::string> args)
        : director_(std::move(director)), args_(std::move(args)) {
        distance_ = director_->default_camera_distance();
        // Verification arcs run HEADLESS: pure GPGPU on the same windowless
        // device path sesolver_vkcheck uses -- no window, no surface, no
        // swapchain, no ImGui. Presenting would tie the loop to vsync AND to
        // Windows' occlusion throttle (a covered window strangles the sim
        // rate and false-fails every wall-clock verdict). The physics arcs
        // never render at all; --dump-frame renders OFFSCREEN only (the
        // render path is exactly what it verifies).
        for (const std::string& a : args_) {
            if (a.rfind("--selftest-", 0) == 0 ||
                a.rfind("--dump-frame", 0) == 0) {
                headless_ = true;
            }
            if (a.rfind("--dump-frame", 0) == 0) {
                needs_render_ = true;
            }
        }
        flow_on_ = has_arg("--flow");  // start with streaklines on (Key F)
    }

    // ---- boot -------------------------------------------------------------
    void init() {
        // SES_VK_VALIDATION=1 turns on VK_LAYER_KHRONOS_validation for the
        // WINDOWED render path too -- sesolver_vkcheck is compute-only, so this
        // is the only way to validate dynamic rendering / present / sync2
        // barriers (the marching-cubes-class of bug lives here, not in compute).
        const char* venv = std::getenv("SES_VK_VALIDATION");
        const bool want_validation = (venv != nullptr && venv[0] == '1');
        if (headless_) {
            // Pure GPGPU: the exact device path sesolver_vkcheck exercises.
            // No SDL video, no window, no surface. The renderer initializes
            // only when an arc verifies it (--dump-frame, offscreen).
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
        window_ = SDL_CreateWindow(
            "Electron wavepacket near a hydrogen nucleus", 1024, 768,
            SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE |
                SDL_WINDOW_HIGH_PIXEL_DENSITY);
        if (window_ == nullptr) {
            fatal_shell_error("SDL window", SDL_GetError());
        }

        // The owning device path (create_instance/create_device) -- what
        // sesolver_vkcheck exercises headless, plus surface + swapchain.
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
        // COMPUTE init is deferred into the loop: engine init (VkFFT plan
        // compile) + the radial atom solve block for seconds, and the window
        // must show a presented frame FIRST -- not sit black behind them.
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

    // ---- main loop ----------------------------------------------------------
    int run() {
        last_tick_ = SDL_GetTicks();
        while (!exit_requested_) {
            pump_events();
            if (!headless_) {
                // Acquire EARLY: the FIFO/vsync wait lands here, so the sim
                // batch and the scene render below run during time that used
                // to be dead inside present().
                presenter_.acquire();
            }
            const std::uint64_t now = SDL_GetTicks();
            // Fixed-cadence 16 ms ticks, coalescing after stalls instead of
            // spiraling.
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

            // Deferred compute init: headless arcs go immediately; the
            // windowed app first PRESENTS one frame (clear + panel) so the
            // seconds of engine init + radial solve show a live window, then
            // initializes on the next iteration.
            if (!compute_init_done_ && (headless_ || presented_once_)) {
                director_->init_compute(
                    vk_ctx_, true,
                    ses_shell::query_free_vram_bytes(vk_ctx_.phys_dev));
                compute_init_done_ = true;
                refresh_status();
            }

            if (compute_init_done_) {
                // The compute half runs before any rendering (the engine's
                // offscreen frames must not interleave with a render). Once
                // per frame, even while paused (Key E works paused). The
                // physics arcs skip rendering entirely (pure GPGPU); only
                // the windowed app and --dump-frame draw the scene.
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

            // UI + present. The scene view samples in the presenter's pass;
            // ImGui rides the same pass after the blit.
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplSDL3_NewFrame();
            ImGui::NewFrame();
            if (auto* hy = director_->hydrogen()) {
                app::draw_hydrogen_panel(*this, ui_, *hy);
            } else if (director_->tunnel() != nullptr) {
                app::draw_generic_panel(*this, ui_, {});
            } else {
                app::draw_generic_panel(
                    *this, ui_, {{"Relax to ground (2)", '2'}});
            }
            // Panel controls mutate the director directly, so keep the status
            // block current here -- otherwise a control changed while PAUSED
            // (no tick, no title-dirty) would read stale forever.
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

    // ---- control entry points every scene shares --------------------------
    // Scenario-specific controls/probes live behind director_->hydrogen() /
    // director_->tunnel() (ses.scenario); the panel and the selftest arcs
    // reach them there.
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
    // Cross-section controls are only meaningful over the volume cloud.
    bool cloud_view() const { return director_ && director_->cloud(); }
    // Selftest hook: turn on the cross-section slice sheet (z-normal, through
    // the nucleus) so lobe signs read clearly. Owns shell UI state.
    void enable_cross_section_demo() {
        ui_.slice_on = true;   // slice sheet ONLY (isolates the slice pass)
        ui_.slice_axis = 2;
        ui_.slice_map = 0;     // density
    }
    void set_time_scale(int scale) {
        director_->set_time_scale(scale);
        refresh_status();
    }
    // Panel readout: achieved au/s and the 1x baseline (62.5 ticks/s x dt)
    // it is measured against.
    double sim_rate() const { return perf_sim_rate_; }
    double baseline_sim_rate() const {
        return (1000.0 / kTickMs) * director_->sim_dt();
    }

    // ImGui panel entry point: feed a scenario key as if typed.
    void press(char ch) {
        if (director_->handle_key(ch)) {
            refresh_status();
        }
    }

    // ---- selftest / verification hooks --------------------------------------
    // The scenario itself + its capability seams (single accessors): the arcs
    // call e.g. hy()->channel_a().
    ses_shell::ScenarioDirector& director() { return *director_; }
    ses_shell::HydrogenApi* hy() { return director_->hydrogen(); }
    ses_shell::TunnelApi* tn() { return director_->tunnel(); }
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
        // ImGui 1.92's Vulkan backend allocates SAMPLED_IMAGE + SAMPLER
        // descriptors (not COMBINED_IMAGE_SAMPLER); DescriptorPoolSize > 0 with
        // DescriptorPool left null lets the backend own a correctly typed pool
        // (and destroy it in ImGui_ImplVulkan_Shutdown).
        info.DescriptorPoolSize = 16;
        // Dynamic rendering (the present pass uses no VkRenderPass): the UI
        // pipeline declares the swapchain colour FORMAT. ImGui 1.92 deep-copies
        // the format array during Init, so a local is safe.
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
                case SDL_EVENT_MOUSE_MOTION:
                    if (!io.WantCaptureMouse &&
                        (e.motion.state & SDL_BUTTON_LMASK) != 0) {
                        azimuth_ -= 0.01 * e.motion.xrel;
                        elevation_ += 0.01 * e.motion.yrel;
                        elevation_ = std::clamp(elevation_, -1.5, 1.5);
                    }
                    break;
                case SDL_EVENT_MOUSE_WHEEL:
                    if (!io.WantCaptureMouse) {
                        // Per-notch zoom: SDL delivers ~1 unit/notch; the
                        // 120 factor sets a brisk step (~11% per notch).
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

    // Generic keys live here; everything else is offered to the scenario as
    // a plain ASCII key (hydrogen: 2/3/4/5/D/E/L).
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
                // Probability-current flow particles (Bohmian tracers).
                flow_on_ = !flow_on_;
                return;
            case SDLK_TAB:
                toggle_view_mode();
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

    // The DRAW half, in ses_vk: assemble FrameInput from the director,
    // resize the offscreen target to the window's pixel size, record the
    // passes (render() is synchronous -- the presenter samples afterwards).
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
        // Probability-flow particles (Key F): drawn over the cloud, frozen
        // while paused so a still frame can still accumulate.
        in.flow = flow_on_ && in.cloud;
        in.flow_animate = !paused_;
        // Cross-section planes (Cloud view display state, owned by ui_).
        in.clip_on = ui_.clip_on;
        in.clip_axis = ui_.clip_axis;
        in.clip_sign = ui_.clip_sign;
        in.clip_offset = ui_.clip_offset;
        in.slice_on = ui_.slice_on;
        in.slice_axis = ui_.slice_axis;
        in.slice_offset = ui_.slice_offset;
        in.slice_map = ui_.slice_map;
        // A single scalar that changes whenever any plane control moves, so
        // the accumulation early-out treats a plane edit as "not static".
        const double plane_tag =
            (ui_.clip_on ? 1.0 : 0.0) + 2.0 * ui_.clip_axis +
            8.0 * ui_.clip_sign + 100.0 * ui_.clip_offset +
            (ui_.slice_on ? 1000.0 : 0.0) + 4000.0 * ui_.slice_axis +
            200000.0 * ui_.slice_offset + 30000000.0 * ui_.slice_map;
        // Temporal accumulation: keep averaging only while NOTHING changed
        // (camera, display params, flash, psi volume, animating particles).
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
        // The psi display volume: the engine's bridge image on the GPU path;
        // null lets the renderer fall back to its CPU-staged texture.
        in.psi_volume = director_->psi_volume_view();
        in.flow_velocity = director_->flow_velocity_view();
        if (in.cloud) {
            if (director_->take_volume_dirty()) {
                // CPU staging only: until compute init has been ATTEMPTED the
                // 268 MB fallback texture must not be allocated (it would be
                // orphaned and would deflate the VRAM-budget probe).
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

    // The context is declared FIRST: members destroy in reverse order, and
    // everything below allocates from its VMA allocator, so the context must
    // outlive them all.
    ses_vk::DeviceContext vk_ctx_;
    ses_vk::SceneRenderer vk_renderer_;
    ses_shell::SwapchainPresenter presenter_;

    std::unique_ptr<ses_shell::ScenarioDirector> director_;

    SDL_Window* window_ = nullptr;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;

    std::vector<std::string> args_;
    app::Scheduler sched_;
    app::UiState ui_;
    std::string status_text_;
    double perf_sim_rate_ = 0.0;     // achieved au/s, ~1 s rolling window
    double perf_last_sim_t_ = 0.0;
    std::uint64_t perf_last_ms_ = 0;
    bool headless_ = false;      // --selftest-*/--dump-frame: pure GPGPU,
                                 // no window/surface/swapchain/ImGui
    bool needs_render_ = false;  // --dump-frame: offscreen renders required
    bool compute_init_done_ = false;  // director init deferred into the loop
    bool presented_once_ = false;     // a live frame exists (gates the init)
    bool exit_requested_ = false;
    int exit_code_ = 0;
    std::uint64_t last_tick_ = 0;

    // Temporal-accumulation bookkeeping.
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
    double distance_ = 150.0;  // frames ~+-62 Bohr at 45 deg fovy: the n<=6
                               // manifold body
};

}  // namespace

int main(int argc, char* argv[]) {
    // GUI-subsystem stderr through a redirect is a fully buffered pipe; a
    // crash then eats every diagnostic. Unbuffered keeps them honest.
    std::setvbuf(stderr, nullptr, _IONBF, 0);

    std::vector<std::string> args{argv, argv + argc};

    // Scenario selection (--scene=hydrogen|harmonic|tunnel); the selftest
    // arcs all drive the hydrogen scene except tunnel, which forces its own.
    std::string scene = "hydrogen";
    for (const std::string& a : args) {
        if (a.rfind("--scene=", 0) == 0) {
            scene = a.substr(8);
        }
    }
    if (std::find(args.begin(), args.end(), "--selftest-tunnel") !=
        args.end()) {
        scene = "tunnel";  // the arc drives its own scene
    } else {
        // Every other selftest arc drives the hydrogen scene (it reaches the
        // director through hydrogen()); force it so a mismatched --scene
        // cannot leave a hydrogen arc dereferencing a null capability. Plain
        // --dump-frame (no --selftest-) keeps the requested scene.
        for (const std::string& a : args) {
            if (a.rfind("--selftest-", 0) == 0) {
                scene = "hydrogen";
                break;
            }
        }
    }
    std::unique_ptr<ses_shell::ScenarioDirector> director;
    if (scene == "tunnel") {
        director = std::make_unique<ses_shell::TunnelingDirector>();
    } else if (scene == "harmonic") {
        director = std::make_unique<ses_shell::HarmonicDirector>();
    } else {
        if (scene != "hydrogen") {
            std::fprintf(stderr, "scene: unknown '%s' -- using hydrogen\n",
                         scene.c_str());
        }
        director = std::make_unique<ses_shell::HydrogenDirector>();
    }

    Shell shell{std::move(director), std::move(args)};
    shell.init();

    // Verification + selftest arcs (--dump-frame*, --selftest-*): registered
    // from ses.scenario.selftest_arcs so main() stays a shell.
    ses_shell::register_verification_arcs(&shell);

    const int code = shell.run();
    shell.shutdown();
    return code;
}
