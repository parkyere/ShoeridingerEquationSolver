// Humble Object shell -- the SDL3 boundary. SDL provides the window, input,
// and the Vulkan surface; the shell OWNS the device (DeviceContext::create),
// the swapchain (vk_present.hpp), and the main loop; Dear ImGui draws the
// control panel + status readout inside the presenter's pass. Everything the
// demo IS lives behind ses_shell::ScenarioDirector (--scene= picks the
// implementation). NO domain logic lives here (docs/ARCHITECTURE.md).
//
// Controls: drag = orbit, wheel = zoom, space = pause, Tab = cloud/surface,
// 1 = real time, 2 = relax (imaginary time), 3 = relax to 2p, 4 = relax to
// 2s, 5 = excite an n=3 state (cascade demo), R = reset, M = measure
// position, E = measure energy, D = decay off/on, L = laser (off -> Z -> X
// -> off), F = flow particles, [ ] = thinner/denser cloud.

// ses_vk first: volk (inside) defines VK_NO_PROTOTYPES and must own the
// vulkan.h inclusion before SDL/ImGui pull their own Vulkan declarations.
// This TU carries the app's VMA implementation (vkcheck has its own; the old
// Qt shell rode the copy embedded in static Qt's QRhi backend).
#define VMA_IMPLEMENTATION
#include "vk_blobs.hpp"

#include <blit_frag_spv.h>
#include <blit_vert_spv.h>

#include "harmonic_director.hpp"
#include "hydrogen_director.hpp"
#include "imgui_ui.hpp"
#include "scheduler.hpp"
#include "selftest_arcs.hpp"
#include "tunneling_director.hpp"
#include "vk_present.hpp"
#include "vram_probe.hpp"

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <SDL3/SDL_main.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fatal_shell_error(const char* stage, const char* detail) {
    std::fprintf(stderr, "%s: %s\n", stage, detail);
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, stage, detail, nullptr);
    std::exit(EXIT_FAILURE);
}

constexpr std::uint64_t kTickMs = 16;

// The shell: window + input + device + presentation + the main loop. The
// same wrapper surface the Qt Viewport exposed, shared by the keyboard, the
// ImGui panel (imgui_ui.hpp), and the selftest arcs (selftest_arcs.hpp).
class Shell {
public:
    Shell(std::unique_ptr<ses_shell::ScenarioDirector> director,
          std::vector<std::string> args)
        : director_(std::move(director)), args_(std::move(args)) {
        hydrogen_ = dynamic_cast<ses_shell::HydrogenDirector*>(director_.get());
        tunneling_ = dynamic_cast<ses_shell::TunnelingDirector*>(director_.get());
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
    }

    // ---- boot -------------------------------------------------------------
    void init() {
        if (headless_) {
            // Pure GPGPU: the exact device path sesolver_vkcheck exercises.
            // No SDL video, no window, no surface. The renderer initializes
            // only when an arc verifies it (--dump-frame, offscreen).
            if (!SDL_Init(SDL_INIT_EVENTS)) {  // SDL_GetTicks + event drain
                fatal_shell_error("SDL init", SDL_GetError());
            }
            if (vk_ctx_.create(false) != ses_vk::Boot::ok) {
                fatal_shell_error("Vulkan device",
                                  "no Vulkan runtime on this machine");
            }
            std::fprintf(stderr, "vk: device %s (headless)\n",
                         vk_ctx_.device_name);
            if (needs_render_ &&
                !vk_renderer_.initialize(vk_ctx_, director_->grid(),
                                         ses_shell::app_render_blobs())) {
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
        if (vk_ctx_.create_instance(false, exts) != ses_vk::Boot::ok) {
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
        std::fprintf(stderr, "vk: device %s\n", vk_ctx_.device_name);

        if (!vk_renderer_.initialize(vk_ctx_, director_->grid(),
                                     ses_shell::app_render_blobs())) {
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
        // must show a presented frame FIRST (the Qt shell's paint order) --
        // not sit black behind them.
    }

    void shutdown() {
        vkDeviceWaitIdle(vk_ctx_.device);
        if (!headless_) {
            ImGui_ImplVulkan_Shutdown();
            ImGui_ImplSDL3_Shutdown();
            ImGui::DestroyContext();
            if (imgui_pool_ != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(vk_ctx_.device, imgui_pool_, nullptr);
                imgui_pool_ = VK_NULL_HANDLE;
            }
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
            // Fixed-cadence ticks (the Qt shell's 16 ms QTimer), coalescing
            // after stalls instead of spiraling.
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
            // initializes on the next iteration (the Qt shell's paint order).
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

            // UI + present. The scene view samples in the presenter's pass;
            // ImGui rides the same pass after the blit.
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplSDL3_NewFrame();
            ImGui::NewFrame();
            if (hydrogen_ != nullptr) {
                ses_shell::draw_hydrogen_panel(*this, ui_);
            } else if (tunneling_ != nullptr) {
                ses_shell::draw_generic_panel(*this, ui_, {});
            } else {
                ses_shell::draw_generic_panel(
                    *this, ui_, {{"Relax to ground (2)", '2'}});
            }
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

    // ---- control entry points (keyboard, ImGui panel, selftest arcs) -------
    void toggle_pause() { paused_ = !paused_; }
    void set_real_time() {
        director_->set_real_time();
        refresh_status();
    }
    void set_relaxing() {
        if (hydrogen_) hydrogen_->set_relaxing();
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
    void measure_energy_now() {
        if (hydrogen_) hydrogen_->measure_energy_now();
    }
    void toggle_view_mode() {
        director_->toggle_view_mode();
        refresh_status();
    }
    void relax_to_excited() {
        if (hydrogen_) hydrogen_->relax_to_excited();
        refresh_status();
    }
    void relax_to_2s() {
        if (hydrogen_) hydrogen_->relax_to_2s();
        refresh_status();
    }
    void toggle_decay() {
        if (hydrogen_) hydrogen_->toggle_decay();
        refresh_status();
    }
    void excite_n3() {
        if (hydrogen_) hydrogen_->excite_n3();
        refresh_status();
    }
    void toggle_laser() {
        if (hydrogen_) hydrogen_->toggle_laser();
        refresh_status();
    }
    void set_efield_e0(double e0) {
        if (hydrogen_) hydrogen_->set_efield_e0(e0);
        refresh_status();
    }
    void set_bfield_b(double b) {
        if (hydrogen_) hydrogen_->set_bfield_b(b);
        refresh_status();
    }
    void toggle_bfield_axis() {
        if (hydrogen_) hydrogen_->toggle_bfield_axis();
        refresh_status();
    }
    int bfield_axis() const { return hydrogen_ ? hydrogen_->bfield_axis() : 2; }
    void set_time_scale(int scale) {
        director_->set_time_scale(scale);
        refresh_status();
    }

    // ImGui panel entry point: feed a scenario key as if typed.
    void press(char ch) {
        if (director_->handle_key(ch)) {
            refresh_status();
        }
    }

    // ---- selftest / verification hooks --------------------------------------
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
        refresh_status();
    }
    double probe_population(int idx) {
        return hydrogen_ ? hydrogen_->probe_population(idx) : 0.0;
    }
    double tunnel_transmitted_max() const {
        return tunneling_ ? tunneling_->transmitted_max() : 0.0;
    }
    void debug_set_camera_distance(double d) {
        distance_ = std::clamp(d, 4.0, 300.0);
    }

    ses_shell::Scheduler& sched() { return sched_; }
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

        const VkDescriptorPoolSize size{
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16};
        VkDescriptorPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pi.maxSets = 16;
        pi.poolSizeCount = 1;
        pi.pPoolSizes = &size;
        if (vkCreateDescriptorPool(vk_ctx_.device, &pi, nullptr,
                                   &imgui_pool_) != VK_SUCCESS) {
            fatal_shell_error("render resources",
                              "ImGui descriptor pool create failed");
        }
        ImGui_ImplVulkan_InitInfo info{};
        info.ApiVersion = VK_API_VERSION_1_1;
        info.Instance = vk_ctx_.instance;
        info.PhysicalDevice = vk_ctx_.phys_dev;
        info.Device = vk_ctx_.device;
        info.QueueFamily = vk_ctx_.queue_family;
        info.Queue = vk_ctx_.queue;
        info.DescriptorPool = imgui_pool_;
        info.PipelineInfoMain.RenderPass = presenter_.render_pass();
        info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
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
                        // One notch = Qt's angleDelta 120 (same zoom rate).
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

    // The context is declared FIRST: members destroy in reverse order, and
    // everything below allocates from its VMA allocator, so the context must
    // outlive them all.
    ses_vk::DeviceContext vk_ctx_;
    ses_vk::SceneRenderer vk_renderer_;
    ses_shell::SwapchainPresenter presenter_;

    std::unique_ptr<ses_shell::ScenarioDirector> director_;
    ses_shell::HydrogenDirector* hydrogen_ = nullptr;
    ses_shell::TunnelingDirector* tunneling_ = nullptr;

    SDL_Window* window_ = nullptr;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkDescriptorPool imgui_pool_ = VK_NULL_HANDLE;

    std::vector<std::string> args_;
    ses_shell::Scheduler sched_;
    ses_shell::UiState ui_;
    std::string status_text_;
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
    } acc_prev_;
    long long frame_index_ = 0;
    bool flow_on_ = false;  // Key F: probability-current flow particles
    bool paused_ = false;
    double absorbance_ = 0.68;  // lightened from 1.5 (default was too opaque)

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
    // from selftest_arcs.hpp so main() stays a shell.
    ses_shell::register_verification_arcs(&shell);

    const int code = shell.run();
    shell.shutdown();
    return code;
}
