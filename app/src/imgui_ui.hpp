#pragma once

// The discoverable-controls panel, in Dear ImGui -- what shell_ui.hpp's Qt
// toolbar was. Buttons mirror the hotkeys (which stay live: the shell only
// routes keys here when ImGui wants capture); the two field sliders drive the
// proper Hamiltonian terms; the director's title_text() readout is rendered
// as a wrapped status block (it long outgrew a window title). Templated on
// the shell type so main() stays a shell (same pattern as selftest_arcs.hpp).

#include <imgui.h>

#include <initializer_list>
#include <string>
#include <utility>

namespace ses_shell {

// Shared UI state the sliders edit between frames (owned by the shell).
struct UiState {
    float efield = 0.0f;    // au; 0 = off (max 0.1)
    float bfield = 0.0f;    // au; 0 = off (max 0.2)
    int time_scale = 1;     // steps-per-frame multiplier (dt untouched)
};

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
void draw_hydrogen_panel(ShellT& shell, UiState& ui) {
    ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse);
    draw_perf_readout(shell);

    if (ImGui::Button("Measure (M)")) shell.measure_now();
    ImGui::SameLine();
    if (ImGui::Button("Measure E (E)")) shell.measure_energy_now();
    ImGui::SameLine();
    if (ImGui::Button("Reset (R)")) shell.reset_simulation();
    ImGui::SameLine();
    if (ImGui::Button("Pause (Space)")) shell.toggle_pause();

    if (ImGui::Button("Real time (1)")) shell.set_real_time();
    ImGui::SameLine();
    if (ImGui::Button("Relax 1s (2)")) shell.set_relaxing();
    ImGui::SameLine();
    if (ImGui::Button("Relax 2p (3)")) shell.relax_to_excited();
    ImGui::SameLine();
    if (ImGui::Button("Relax 2s (4)")) shell.relax_to_2s();
    ImGui::SameLine();
    if (ImGui::Button("Excite n=3/4 (5)")) shell.excite_n3();

    if (ImGui::Button("Decay (D)")) shell.toggle_decay();
    ImGui::SameLine();
    if (ImGui::Button("Laser (L)")) shell.toggle_laser();
    ImGui::SameLine();
    if (ImGui::Button("Cloud/Surface (Tab)")) shell.toggle_view_mode();

    // Static E-field (+z): 0 = off, 0.1 au full scale (Stark; field-ionizes
    // above ~0.03 au from the ground state).
    if (ImGui::SliderFloat("E-field +z (au)", &ui.efield, 0.0f, 0.1f, "%.3f")) {
        shell.set_efield_e0(static_cast<double>(ui.efield));
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
    const int axis = shell.bfield_axis();
    const char* axis_name = axis == 2 ? "z" : (axis == 0 ? "x" : "y");
    if (ImGui::Button(axis_name)) {
        shell.toggle_bfield_axis();
    }
    ImGui::SameLine();
    if (ImGui::SliderFloat("B-field (au)", &ui.bfield, 0.0f, 0.2f, "%.3f")) {
        shell.set_bfield_b(static_cast<double>(ui.bfield));
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
    bool mcwf = shell.mcwf_damping();
    if (ImGui::Checkbox("MCWF damping", &mcwf)) {
        shell.set_mcwf_damping(mcwf);
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

}  // namespace ses_shell
