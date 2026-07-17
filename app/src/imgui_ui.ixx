module;
#include <imgui.h>
#include <initializer_list>
#include <string>
#include <utility>
export module app.imgui_ui;
export import ses.scenario;  // HydrogenApi (the hydrogen panel's control seam)


// The discoverable-controls panel, in Dear ImGui: a clickable mirror of the
// hotkeys. Buttons mirror the hotkeys (which stay live: the shell only
// routes keys here when ImGui wants capture); the two field sliders drive the
// proper Hamiltonian terms; the director's status_text() readout renders as a
// wrapped status block. Templated on the shell type so main() stays a shell.


export namespace app {

// Shared UI state the sliders edit between frames (owned by the shell).
struct UiState {
    float efield = 0.0f;    // au; 0 = off (max 0.1)
    float bfield = 0.0f;    // au; 0 = off (max 0.2)
    int time_scale = 1;     // steps-per-frame multiplier (dt untouched)
    // Cross-section planes (Cloud view). axis 0/1/2 = x/y/z normal.
    bool clip_on = false;
    int clip_axis = 2;
    int clip_sign = 1;       // +1 hides the +normal half, -1 the other
    float clip_offset = 0.0f;   // Bohr; 0 = through the nucleus
    bool slice_on = false;
    int slice_axis = 2;
    float slice_offset = 0.0f;
    int slice_map = 0;       // 0 density, 1 Re(psi), 2 phase
    // 1D harmonic scene: live well stiffness (matches kHo1dOmega at boot;
    // UiState resets on scene switch, so the slider and director agree).
    float ho_omega = 0.25f;
};

// The x/y/z axis-cycle button shared by the cross-section controls.
inline void draw_axis_cycle(const char* id, int& axis) {
    const char* name = axis == 0 ? "x" : (axis == 1 ? "y" : "z");
    ImGui::PushID(id);
    if (ImGui::Button(name)) {
        axis = (axis + 1) % 3;
    }
    ImGui::PopID();
}

// The cross-section section (Cloud view only): a clip plane that cuts the
// cloud open and a slice sheet that paints psi on the plane.
template <typename ShellT>
void draw_cross_section(ShellT& shell, UiState& ui) {
    (void)shell;
    const double box = 80.0;  // Bohr half-extent (matches the +-80 grid)
    ImGui::Checkbox("Clip plane", &ui.clip_on);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Cut the cloud open along a plane through the "
                          "nucleus to see the interior.");
    }
    if (ui.clip_on) {
        ImGui::SameLine();
        draw_axis_cycle("clipax", ui.clip_axis);
        ImGui::SameLine();
        if (ImGui::Button(ui.clip_sign > 0 ? "side +" : "side -")) {
            ui.clip_sign = -ui.clip_sign;
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(140.0f);
        ImGui::SliderFloat("clip @", &ui.clip_offset,
                           static_cast<float>(-box), static_cast<float>(box),
                           "%.0f a0");
    }
    ImGui::Checkbox("Slice", &ui.slice_on);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Paint psi on a plane through the nucleus -- a "
                          "cross-section sheet. Nodal planes show as dark "
                          "bands.");
    }
    if (ui.slice_on) {
        ImGui::SameLine();
        draw_axis_cycle("sliceax", ui.slice_axis);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120.0f);
        ImGui::Combo("map", &ui.slice_map, "Density\0Re(psi)\0Phase\0");
        ImGui::SetNextItemWidth(220.0f);
        ImGui::SliderFloat("slice @", &ui.slice_offset,
                           static_cast<float>(-box), static_cast<float>(box),
                           "%.0f a0");
    }
}

// Scene picker, shared by every scenario panel: the three demos, switchable
// live -- the shell swaps the director with the device idle and re-runs the
// deferred compute init (the same path boot uses), so the window never
// blocks. Switching to Hydrogen replays its startup atlas build.
template <typename ShellT>
void draw_scene_picker(ShellT& shell) {
    int cur = shell.scene_index();
    ImGui::SetNextItemWidth(200.0f);
    if (ImGui::Combo("Scene", &cur,
                     "Hydrogen atom\0Harmonic trap\0Tunneling barrier\0"
                     "1D harmonic ladder\0"
                     "1D tunneling barrier\0")) {
        shell.request_scene(cur);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Switch the demo live. Hydrogen re-runs its atlas "
                          "build; the trap and the barrier start instantly.");
    }
    ImGui::Separator();
}

// Performance readout, shared by every scenario panel: rendering fps (ImGui's
// rolling average) and the ACHIEVED simulated-time rate with its multiple of
// the 1x baseline -- the honest counterpart of the time-scale slider (the
// multiple saturates below the slider once the GPU runs out of headroom).
template <typename ShellT>
void draw_perf_readout(ShellT& shell) {
    const double rate = shell.sim_rate();
    const double base = shell.baseline_sim_rate();
    ImGui::Text("%.1f fps   %.2f au/s%s", ImGui::GetIO().Framerate, rate,
                base > 0.0 ? "" : " (warming up)");
    if (base > 0.0) {
        ImGui::SameLine();
        ImGui::Text("(x%.1f)", rate / base);
    }
    ImGui::Separator();
}

// The visualized-time slider, shared by every scenario panel: multiplies the
// integrator steps per rendered frame (dt and accuracy untouched); past the
// GPU's headroom the fps drops honestly instead of skipping physics.
template <typename ShellT>
void draw_time_scale(ShellT& shell, UiState& ui) {
    if (ImGui::SliderInt("Time scale", &ui.time_scale, 1, 16, "x%d")) {
        shell.set_time_scale(ui.time_scale);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Run N integrator steps per rendered frame (same "
                          "dt -- accuracy is unchanged).\nWhen the GPU "
                          "saturates, the frame rate drops instead.");
    }
}

template <typename ShellT>
void draw_hydrogen_panel(ShellT& shell, UiState& ui, ses_shell::HydrogenApi& hy) {
    ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse);
    draw_scene_picker(shell);
    draw_perf_readout(shell);

    if (ImGui::Button("Measure (M)")) shell.measure_now();
    ImGui::SameLine();
    // Honest label: sampling a SINGLE eigenstate is the maximal (n,l,m)
    // measurement, not a bare energy measurement (2s vs 2p are degenerate).
    if (ImGui::Button("Measure nlm (E)")) hy.measure_energy_now();
    ImGui::SameLine();
    if (ImGui::Button("Reset (R)")) shell.reset_simulation();
    ImGui::SameLine();
    if (ImGui::Button("Pause (Space)")) shell.toggle_pause();

    // Partial projective measurements: one quantum number, one degenerate
    // subspace -- superpositions inside it survive the collapse.
    if (ImGui::Button("Measure n")) hy.measure_n_shell_now();
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("True energy measurement: project onto one n "
                          "SHELL.\nDegenerate l,m superpositions inside the "
                          "shell survive (2s+2p stays 2s+2p).");
    }
    ImGui::SameLine();
    if (ImGui::Button("Measure l")) hy.measure_l_now();
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Measure L^2: project onto one l.\nSuperpositions "
                          "across n and m survive (partial collapse).");
    }
    ImGui::SameLine();
    if (ImGui::Button("Measure m")) hy.measure_m_now();
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Measure L_z: project onto one signed m.\nA p_x "
                          "lobe collapses to a rotating |m|=1 ring -- and a "
                          "repeat measurement agrees with certainty.");
    }

    if (ImGui::Button("Real time (1)")) shell.set_real_time();
    ImGui::SameLine();
    if (ImGui::Button("Relax 1s (2)")) hy.set_relaxing();
    ImGui::SameLine();
    if (ImGui::Button("Relax 2p (3)")) hy.relax_to_excited();
    ImGui::SameLine();
    if (ImGui::Button("Relax 2s (4)")) hy.relax_to_2s();
    ImGui::SameLine();
    if (ImGui::Button("Excite n=3/4 (5)")) hy.excite_n3();

    if (ImGui::Button("Decay (D)")) hy.toggle_decay();
    ImGui::SameLine();
    if (ImGui::Button("Laser (L)")) hy.toggle_laser();
    ImGui::SameLine();
    if (ImGui::Button("Cloud/Surface (Tab)")) shell.toggle_view_mode();

    // Static E-field (+z): 0 = off, 0.1 au full scale (Stark; field-ionizes
    // above ~0.03 au from the ground state).
    if (ImGui::SliderFloat("E-field +z (au)", &ui.efield, 0.0f, 0.1f, "%.3f")) {
        hy.set_efield_e0(static_cast<double>(ui.efield));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Static uniform E-field along +z (Stark).\n"
                          "0 = off; full = 0.1 au (~5.1e10 V/m).\nThe 1s "
                          "barely moves below ~0.03 au, then field-ionizes.");
    }
    if (ui.efield > 0.0f) {
        ImGui::SameLine();
        ImGui::Text("%.1e V/m", ui.efield * 5.14220674e11);
    }

    // Magnetic field: axis cycle (z -> x -> y) + strength; psi evolves under
    // the proper minimal-coupling Hamiltonian and precesses at omega = B/2.
    const int axis = hy.bfield_axis();
    const char* axis_name = axis == 2 ? "z" : (axis == 0 ? "x" : "y");
    if (ImGui::Button(axis_name)) {
        hy.toggle_bfield_axis();
    }
    ImGui::SameLine();
    if (ImGui::SliderFloat("B-field (au)", &ui.bfield, 0.0f, 0.2f, "%.3f")) {
        hy.set_bfield_b(static_cast<double>(ui.bfield));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Magnetic field along the chosen axis. The cloud "
                          "precesses (Larmor) at omega = B/2.\nPrepare a p_x / "
                          "d_xy state to see it rotate; s and p_z do not "
                          "precess about z.");
    }

    draw_time_scale(shell, ui);

    // MCWF no-jump damping: superpositions visibly emit between jumps
    // (H_eff amplitude decay); pure eigenstates are unaffected either way.
    bool mcwf = hy.mcwf_damping();
    if (ImGui::Checkbox("MCWF damping", &mcwf)) {
        hy.set_mcwf_damping(mcwf);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Between jumps, drain excited amplitudes as "
                          "e^(-gamma t/2) conditioned on no photon\n"
                          "(Monte Carlo wave function). Watch a superposition's "
                          "beat fade as it emits.\nAuto-skipped while the laser "
                          "drives (the accelerated gammas would swamp the "
                          "flop).");
    }

    ImGui::Separator();
    // Cross-section planes: only meaningful over the volume cloud.
    if (shell.cloud_view()) {
        draw_cross_section(shell, ui);
        ImGui::Separator();
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(shell.status_text().c_str());
    ImGui::PopTextWrapPos();
    ImGui::End();
}

template <typename ShellT>
void draw_generic_panel(ShellT& shell, UiState& ui,
                        std::initializer_list<std::pair<const char*, char>>
                            scene_keys) {
    ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse);
    draw_scene_picker(shell);
    draw_perf_readout(shell);
    if (ImGui::Button("Real time (1)")) shell.set_real_time();
    ImGui::SameLine();
    for (const auto& k : scene_keys) {
        if (ImGui::Button(k.first)) shell.press(k.second);
        ImGui::SameLine();
    }
    if (ImGui::Button("Measure (M)")) shell.measure_now();
    if (ImGui::Button("Reset (R)")) shell.reset_simulation();
    ImGui::SameLine();
    if (ImGui::Button("Cloud/Surface (Tab)")) shell.toggle_view_mode();
    ImGui::SameLine();
    if (ImGui::Button("Pause (Space)")) shell.toggle_pause();
    draw_time_scale(shell, ui);
    ImGui::Separator();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(shell.status_text().c_str());
    ImGui::PopTextWrapPos();
    ImGui::End();
}

// The 1D harmonic-ladder panel: ladder/superposition controls + the live
// well-stiffness slider (a sudden quench -- psi is kept; the director
// re-derives the spectral-band ladder cap).
template <typename ShellT>
void draw_ladder1d_panel(ShellT& shell, UiState& ui, ses_shell::Ladder1dApi& ld) {
    ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse);
    draw_scene_picker(shell);
    draw_perf_readout(shell);
    if (ImGui::Button("Ladder up (U)")) shell.press('U');
    ImGui::SameLine();
    if (ImGui::Button("Ladder down (D)")) shell.press('D');
    ImGui::SameLine();
    if (ImGui::Button("Random mix (S)")) shell.press('S');
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Random coherent superposition over n = 0..5 (a "
                          "pure state).\nThe phasor curve sloshes -- watch "
                          "the ladder act on EVERY component.");
    }
    ImGui::SameLine();
    if (ImGui::Button("Ground (2)")) shell.press('2');
    if (ImGui::Button("Reset (R)")) shell.reset_simulation();
    ImGui::SameLine();
    if (ImGui::Button("Pause (Space)")) shell.toggle_pause();
    if (ImGui::SliderFloat("Well omega (au)", &ui.ho_omega, 0.05f, 1.0f,
                           "%.2f")) {
        ld.set_omega(static_cast<double>(ui.ho_omega));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Well stiffness w (width ~ 1/sqrt(w)).\nChanging "
                          "it is a sudden QUENCH: psi is kept and breathes "
                          "in\nthe new well; the ladder cap adapts (now "
                          "n <= %d).",
                          ld.max_level());
    }
    draw_time_scale(shell, ui);
    ImGui::Separator();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(shell.status_text().c_str());
    ImGui::PopTextWrapPos();
    ImGui::End();
}

}  // namespace app
