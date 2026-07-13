#pragma once

// The scenario seam: everything a demo IS, behind one Qt-free interface.
// The Qt shell owns exactly one ScenarioDirector (chosen by --scene=) and
// talks to it through this contract; scenario-specific keys go through
// handle_key (plain ASCII, shell translates Qt codes). No Qt type crosses.

#include "vk_device.hpp"

#include <core/colormap.hpp>
#include <core/grid.hpp>
#include <core/marching_cubes.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ses_shell {

class ScenarioDirector {
public:
    virtual ~ScenarioDirector() = default;

    // ---- lifecycle ----
    virtual const ses::Grid3D& grid() const = 0;
    virtual void init_compute(ses_vk::DeviceContext& ctx, bool device_ok,
                              std::int64_t free_vram_bytes) = 0;
    virtual void release_gpu() = 0;
    virtual bool compute_attempted() const = 0;
    virtual bool gpu_ok() const = 0;

    // ---- per-frame / per-tick ----
    virtual void run_frame() = 0;
    virtual void tick() = 0;

    // ---- controls every scenario supports ----
    virtual void set_real_time() = 0;
    virtual void reset_simulation() = 0;
    virtual void measure_now() = 0;
    virtual void toggle_view_mode() = 0;
    // Scenario-specific keys (upper-case letters / digits); true = handled.
    virtual bool handle_key(char key) = 0;

    // ---- state the shell gates on ----
    virtual bool solving() const = 0;      // startup solve owns the GPU state
    virtual bool scene_ready() const = 0;  // demo fully armed (selftest gate)

    // Camera start distance framing this scene's box (Bohr).
    virtual double default_camera_distance() const { return 150.0; }

    // Visualized time scale: multiply the steps SUPPLIED per wall tick (and
    // the per-frame consumption cap). dt is untouched -- more integrator
    // steps per rendered frame, never larger ones -- so accuracy is
    // preserved and the GPU saturating just lowers fps honestly.
    virtual void set_time_scale(int scale) { (void)scale; }
    virtual int time_scale() const { return 1; }

    // ---- display accessors (FrameInput assembly + title) ----
    virtual bool cloud() const = 0;
    virtual double peak() const = 0;
    virtual VkImageView psi_volume_view() = 0;
    virtual float next_flash_intensity() = 0;
    virtual bool take_volume_written() = 0;
    virtual bool take_volume_dirty() = 0;
    virtual bool take_mesh_dirty() = 0;
    virtual void mark_display_dirty() = 0;
    virtual bool take_title_dirty() = 0;
    virtual const std::vector<float>& psi_staging() const = 0;
    virtual const ses::Mesh& mesh() const = 0;
    virtual const std::vector<ses::Rgb>& colors() const = 0;
    virtual std::string title_text() = 0;
};

}  // namespace ses_shell
