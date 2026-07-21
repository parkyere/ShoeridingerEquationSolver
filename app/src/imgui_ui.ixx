module;
#include <imgui.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>
export module app.imgui_ui;
export import ses.scenario;


export namespace app {

struct UiState {
    float efield = 0.0f;    // au; 0 = off
    float bfield = 0.0f;    // au; 0 = off
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
    // matches kHo1dOmega at boot (UiState resets per scene -> slider agrees).
    float ho_omega = 0.25f;
    // matches kDw1dBarrier at boot.
    float dw_barrier = 0.12f;
    // snapped to the grid by the director.
    float h2p_r = 2.0f;
    // boot values match the director's.
    float ds_sep = 8.0f;
    float ds_width = 2.0f;
    float ds_flux_pi = 0.0f;
    float la_b = 0.4f;
    float la_k0 = 1.5f;
    float cr_r = 10.0f;
    float qd_w0 = 0.5f;
    float qd_b = 0.6f;
    float bl_v0 = 1.5f;
    float bl_f = 0.05f;
    float ru_e = 25.0f;
    float ru_z = 79.0f;
};

inline void draw_axis_cycle(const char* id, int& axis) {
    const char* name = axis == 0 ? "x" : (axis == 1 ? "y" : "z");
    ImGui::PushID(id);
    if (ImGui::Button(name)) {
        axis = (axis + 1) % 3;
    }
    ImGui::PopID();
}

template <typename ShellT>
void draw_cross_section(ShellT& shell, UiState& ui) {
    (void)shell;
    const double box = 80.0;  // Bohr half-extent (+-80 grid)
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

template <typename ShellT>
void draw_scene_picker(ShellT& shell) {
    int cur = shell.scene_index();
    ImGui::SetNextItemWidth(200.0f);
    // Labels come from the shell's scene table so the combo can never drift
    // out of sync with the registered scenes.
    std::vector<const char*> labels;
    labels.reserve(static_cast<std::size_t>(shell.scene_count()));
    for (int i = 0; i < shell.scene_count(); ++i) {
        labels.push_back(shell.scene_label(i));
    }
    if (ImGui::Combo("Scene", &cur, labels.data(),
                     static_cast<int>(labels.size()))) {
        shell.request_scene(cur);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Switch the demo live. Hydrogen re-runs its atlas "
                          "build; the trap and the barrier start instantly.");
    }
    ImGui::Separator();
}

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

template <typename ShellT>
void draw_time_scale(ShellT& shell, UiState& ui) {
    // Read back director truth: a programmatic change (Real time -> x1) must not leave the slider stale.
    ui.time_scale = shell.time_scale();
    if (ImGui::SliderInt("Time scale", &ui.time_scale, 1, 16, "x%d")) {
        shell.set_time_scale(ui.time_scale);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Run N integrator steps per rendered frame (same "
                          "dt -- accuracy is unchanged).\nWhen the GPU "
                          "saturates, the frame rate drops instead.");
    }
}

// Rainbow is an ENERGY-SCALE decoration, not physical color (most H lines are UV/IR).
inline void draw_spectrometer(ses_shell::HydrogenApi& hy) {
    const double emax = hy.spectro_max_ev();
    if (emax <= 0.0) {
        return;  // not an emission scene (the trap shares this Api)
    }
    const ImGuiIO& io = ImGui::GetIO();
    const float win_w = 190.0f;
    const float win_h = io.DisplaySize.y * 0.78f;
    ImGui::SetNextWindowPos(
        ImVec2(io.DisplaySize.x - win_w - 10.0f, 48.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(win_w, win_h), ImGuiCond_Always);
    ImGui::Begin("Spectrometer", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoMove);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float bx = origin.x + 4.0f;
    const float bw = 30.0f;
    const float by = origin.y + 12.0f;
    const float bh = win_h - 70.0f;
    // Rainbow scale, 7 stops red(bottom) -> violet(top).
    const ImU32 stops[7] = {
        IM_COL32(225, 40, 40, 255),  IM_COL32(235, 140, 35, 255),
        IM_COL32(235, 215, 45, 255), IM_COL32(60, 195, 85, 255),
        IM_COL32(55, 130, 235, 255), IM_COL32(45, 60, 180, 255),
        IM_COL32(150, 60, 220, 255)};
    for (int s = 0; s < 6; ++s) {
        const float y_lo = by + bh * (1.0f - s / 6.0f);
        const float y_hi = by + bh * (1.0f - (s + 1) / 6.0f);
        dl->AddRectFilledMultiColor(ImVec2(bx, y_hi), ImVec2(bx + bw, y_lo),
                                    stops[s + 1], stops[s + 1], stops[s],
                                    stops[s]);
    }
    dl->AddRect(ImVec2(bx - 1, by - 1), ImVec2(bx + bw + 1, by + bh + 1),
                IM_COL32(200, 200, 210, 160));
    char buf[32];
    for (int t = 0; t <= 4; ++t) {
        const double ev = emax * t / 4.0;
        const float y = by + bh * static_cast<float>(1.0 - ev / emax);
        dl->AddLine(ImVec2(bx + bw, y), ImVec2(bx + bw + 4, y),
                    IM_COL32(200, 200, 210, 160));
        std::snprintf(buf, sizeof(buf), "%.1f", ev);
        dl->AddText(ImVec2(bx + bw + 6, y - 7), IM_COL32(180, 180, 190, 200),
                    buf);
    }
    // Emitted lines, deduplicated to 0.01 eV buckets (repeats brighten).
    const int n = hy.spectro_count();
    int cents[64];
    int counts[64];
    int distinct = 0;
    for (int i = 0; i < n; ++i) {
        const int c = static_cast<int>(hy.spectro_ev(i) * 100.0 + 0.5);
        int k = 0;
        while (k < distinct && cents[k] != c) {
            ++k;
        }
        if (k == distinct && distinct < 64) {
            cents[distinct] = c;
            counts[distinct] = 0;
            ++distinct;
        }
        if (k < 64) {
            ++counts[k];
        }
    }
    for (int k = 0; k < distinct; ++k) {
        const double ev = cents[k] / 100.0;
        const float y = by + bh * static_cast<float>(1.0 - ev / emax);
        const int a = counts[k] > 2 ? 255 : 160 + 32 * counts[k];
        dl->AddRectFilled(ImVec2(bx - 3, y - 1),
                          ImVec2(bx + bw + 3, y + 1),
                          IM_COL32(255, 255, 255, a));
        std::snprintf(buf, sizeof(buf), "%.2f eV x%d", ev, counts[k]);
        dl->AddText(ImVec2(bx + bw + 34, y - 7),
                    IM_COL32(255, 255, 255, 230), buf);
    }
    ImGui::Dummy(ImVec2(win_w - 20.0f, bh + 20.0f));
    ImGui::TextDisabled("photons: %d", n);
    ImGui::End();
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
    if (ImGui::Button("Measure nlm (E)")) hy.measure_energy_now();
    ImGui::SameLine();
    if (ImGui::Button("Reset (R)")) shell.reset_simulation();
    ImGui::SameLine();
    if (ImGui::Button("Pause (Space)")) shell.toggle_pause();
    ImGui::SameLine();
    if (ImGui::Button("Face z (Z)")) shell.snap_camera_z();

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
    ImGui::SameLine();
    if (ImGui::Button("Kepler packet (K)")) hy.seed_kepler();
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Circular-state Rydberg packet: the density lobe\n"
                          "orbits the nucleus at the classical Kepler rate\n"
                          "1/n^3, disperses, and partially revives.");
    }

    if (ImGui::Button("Decay (D)")) hy.toggle_decay();
    ImGui::SameLine();
    if (ImGui::Button("Laser (L)")) hy.toggle_laser();
    ImGui::SameLine();
    if (ImGui::Button("Cloud/Surface (Tab)")) shell.toggle_view_mode();

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

    const int axis = hy.bfield_axis();
    const char* axis_name = axis == 2 ? "z" : (axis == 0 ? "x" : "y");
    if (ImGui::Button(axis_name)) {
        hy.toggle_bfield_axis();
    }
    ImGui::SameLine();
    // Director truth first: R / Laser zero B, so the slider must follow (cf. time scale).
    ui.bfield = static_cast<float>(hy.bfield_b());
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
    if (shell.cloud_view()) {
        draw_cross_section(shell, ui);
        ImGui::Separator();
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(shell.status_text().c_str());
    ImGui::PopTextWrapPos();
    ImGui::End();
    draw_spectrometer(hy);
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
    ImGui::SameLine();
    if (ImGui::Button("Face z (Z)")) shell.snap_camera_z();
    draw_time_scale(shell, ui);
    ImGui::Separator();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(shell.status_text().c_str());
    ImGui::PopTextWrapPos();
    ImGui::End();
}

// brightness = sqrt(weight): faint components still read.
template <typename ApiT>
void draw_ho_spectrum_strip(ApiT& api) {
    ImGui::TextUnformatted("State spectrum 0-100 eV (linear combination)");
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float w = ImGui::GetContentRegionAvail().x;
    const float h = 34.0f;
    dl->AddRectFilled(ImVec2(p.x, p.y), ImVec2(p.x + w, p.y + h),
                      IM_COL32(12, 12, 18, 255));
    const int n = api.spectrum_count();
    double wmax = 1e-12;
    for (int i = 0; i < n; ++i) {
        wmax = std::max(wmax, api.spectrum_weight(i));
    }
    for (int i = 0; i < n; ++i) {
        const double ev = api.spectrum_ev(i);
        if (ev > 100.0) {
            continue;
        }
        const float b =
            static_cast<float>(std::sqrt(api.spectrum_weight(i) / wmax));
        if (b < 0.02f) {
            continue;
        }
        const float x = p.x + w * static_cast<float>(ev / 100.0);
        dl->AddLine(ImVec2(x, p.y + 2.0f), ImVec2(x, p.y + h - 2.0f),
                    IM_COL32(90 + static_cast<int>(160.0f * b),
                             220, 255, 40 + static_cast<int>(215.0f * b)),
                    2.0f);
    }
    dl->AddRect(ImVec2(p.x, p.y), ImVec2(p.x + w, p.y + h),
                IM_COL32(70, 70, 90, 255));
    ImGui::Dummy(ImVec2(w, h + 4.0f));
}

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
    if (ImGui::Button("Cat (C)")) shell.press('C');
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Schrodinger cat |a> + |-a>: two coherent lobes.\n"
                          "Interference fringes appear when they cross the\n"
                          "trap center.");
    }
    ImGui::SameLine();
    if (ImGui::Button(ld.loss_on() ? "Loss off (X)" : "Photon loss (X)")) {
        shell.press('X');
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Cavity decay as an MCWF unraveling: each lost\n"
                          "photon FLIPS the cat's fringe parity while <n>\n"
                          "bleeds at kappa -- decoherence, one quantum\n"
                          "trajectory at a time.");
    }
    if (ld.loss_on()) {
        ImGui::SameLine();
        ImGui::Text("photons lost: %lld", ld.jump_count());
    }
    if (ImGui::Button("Reset (R)")) shell.reset_simulation();
    ImGui::SameLine();
    if (ImGui::Button("Pause (Space)")) shell.toggle_pause();
    ImGui::SameLine();
    if (ImGui::Button("Face z (Z)")) shell.snap_camera_z();
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Snap the camera onto the +z axis: the textbook "
                          "straight-on view\n(x right, y up; the phasor "
                          "twist goes into the screen).");
    }
    // Apply on release, not per drag frame: each omega re-sweeps the whole
    // Hermite chain (~19000 levels at the stiff stop) -- too costly at 60 Hz.
    ImGui::SliderFloat("Well omega (au)", &ui.ho_omega, 0.05f, 4.0f, "%.2f");
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        ld.set_omega(static_cast<double>(ui.ho_omega));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Well stiffness w (width ~ 1/sqrt(w)); applies "
                          "when released.\nChanging it is a sudden QUENCH: "
                          "psi is kept and breathes in\nthe new well. The "
                          "ladder cap is the box ceiling (turning points\n"
                          "must fit +-100 Bohr), so it RISES with w -- now "
                          "n <= %d.",
                          ld.max_level());
    }
    draw_ho_spectrum_strip(ld);
    draw_time_scale(shell, ui);
    ImGui::Separator();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(shell.status_text().c_str());
    ImGui::PopTextWrapPos();
    ImGui::End();
}

template <typename ShellT>
void draw_doublewell_panel(ShellT& shell, UiState& ui,
                           ses_shell::DoubleWellApi& dw) {
    ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse);
    draw_scene_picker(shell);
    draw_perf_readout(shell);
    if (ImGui::Button("Reset (R)")) shell.reset_simulation();
    ImGui::SameLine();
    if (ImGui::Button("Pause (Space)")) shell.toggle_pause();
    ImGui::SameLine();
    if (ImGui::Button("Face z (Z)")) shell.snap_camera_z();
    if (ImGui::SliderFloat("Barrier Vb (Ha)", &ui.dw_barrier, 0.04f, 0.30f,
                           "%.2f")) {
        dw.set_barrier(static_cast<double>(ui.dw_barrier));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("dE is exponential in the barrier: raising it "
                          "makes the\nleft<->right oscillation crawl. Moving "
                          "the slider re-prepares\nthe left-well state "
                          "(currently dE = %.2e Ha).",
                          dw.splitting());
    }
    draw_time_scale(shell, ui);
    ImGui::Separator();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(shell.status_text().c_str());
    ImGui::PopTextWrapPos();
    ImGui::End();
}

template <typename ShellT>
void draw_h2plus_panel(ShellT& shell, UiState& ui, ses_shell::MoleculeApi& ml) {
    ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse);
    draw_scene_picker(shell);
    draw_perf_readout(shell);
    const int nmo = ml.state_count();
    ImGui::TextUnformatted("Known orbitals (exact prolate-spheroidal atlas):");
    if (ImGui::BeginListBox("##h2p_orbitals", ImVec2(-1.0f, 160.0f))) {
        for (int k = 0; k < nmo; ++k) {
            char lbl[96];
            std::snprintf(lbl, sizeof(lbl), "%2d  %-18s  E = %.4f Ha", k + 1,
                          ml.orbital_label(k), ml.energy(k));
            if (ImGui::Selectable(lbl)) {
                shell.press(static_cast<char>('2' + k));
            }
        }
        ImGui::EndListBox();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("The known H2+ molecular orbitals, synthesized from "
                          "the exact\nprolate-spheroidal solution (baked "
                          "atlas). Click to show one.");
    }
    if (ImGui::Button("Random wavefunction (S)")) shell.press('S');
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Drop an arbitrary normalized state of random "
                          "shape and watch it evolve\n(a superposition of "
                          "orbitals -- it beats and disperses).");
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset (R)")) shell.reset_simulation();
    if (ImGui::Button("Pause (Space)")) shell.toggle_pause();
    ImGui::SameLine();
    if (ImGui::Button("Face z (Z)")) shell.snap_camera_z();
    // No bond-length slider: R fixed at the physical equilibrium (~2.0 bohr,
    // rigid Born-Oppenheimer) -- a free knob would misrepresent it.
    if (ml.prepared(0)) {
        ImGui::Text("R fixed at equilibrium; E_total(1sigma_g) = %.4f Ha",
                    ml.energy(0) + ml.nuclear_repulsion());
    }
    draw_time_scale(shell, ui);
    ImGui::Separator();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(shell.status_text().c_str());
    ImGui::PopTextWrapPos();
    ImGui::End();
}

template <typename ShellT>
void draw_benzene_panel(ShellT& shell, UiState& ui, ses_shell::MoleculeApi& ml) {
    ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse);
    draw_scene_picker(shell);
    draw_perf_readout(shell);
    if (ImGui::Button("Ground (2)")) shell.press('2');
    ImGui::SameLine();
    if (ImGui::Button("Excited 1 (3)")) shell.press('3');
    ImGui::SameLine();
    if (ImGui::Button("Excited 2 (4)")) shell.press('4');
    ImGui::SameLine();
    if (ImGui::Button("Reset (R)")) shell.reset_simulation();
    if (ImGui::Button("Pause (Space)")) shell.toggle_pause();
    ImGui::SameLine();
    if (ImGui::Button("Face z (Z)")) shell.snap_camera_z();
    if (ml.prepared(0)) {
        ImGui::Text("E0 = %.3f Ha", ml.energy(0));
        if (ml.prepared(1)) {
            ImGui::SameLine();
            ImGui::Text("E1 = %.3f", ml.energy(1));
        }
        if (ml.prepared(2)) {
            ImGui::SameLine();
            ImGui::Text("E2 = %.3f", ml.energy(2));
        }
    }
    draw_time_scale(shell, ui);
    ImGui::Separator();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(shell.status_text().c_str());
    ImGui::PopTextWrapPos();
    ImGui::End();
}

template <typename ShellT>
void draw_doubleslit_panel(ShellT& shell, UiState& ui,
                           ses_shell::SlitApi& sl) {
    ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse);
    draw_scene_picker(shell);
    draw_perf_readout(shell);
    if (ImGui::Button("Fire electron (2)")) shell.press('2');
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("One electron per shot; the screen keeps\n"
                          "accumulating arrivals shot after shot.\n"
                          "Electrons don't interact -- a new shot replaces\n"
                          "the one in flight.");
    }
    ImGui::SameLine();
    if (ImGui::Button("Pause (Space)")) shell.toggle_pause();
    ImGui::SameLine();
    if (ImGui::Button("Face z (Z)")) shell.snap_camera_z();
    if (ImGui::SliderFloat("Slit separation d", &ui.ds_sep, 4.0f, 16.0f,
                           "%.1f")) {
        sl.set_separation(static_cast<double>(ui.ds_sep));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Fringe spacing on the screen goes as 1/d.");
    }
    if (ImGui::SliderFloat("Slit width w", &ui.ds_width, 1.0f, 4.0f,
                           "%.1f")) {
        sl.set_width(static_cast<double>(ui.ds_width));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Single-slit envelope width goes as 1/w.");
    }
    if (ImGui::SliderFloat("Solenoid flux (pi)", &ui.ds_flux_pi, -2.0f, 2.0f,
                           "%.2f")) {
        sl.set_flux(static_cast<double>(ui.ds_flux_pi) * 3.14159265f);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Aharonov-Bohm: a solenoid behind the wall "
                          "between the slits.\nFringes SLIDE, the envelope "
                          "stays -- period 2 pi (one flux quantum).\nB = 0 "
                          "everywhere the electron goes.");
    }
    ImGui::Text("Transmitted: %.1f%%", 100.0 * sl.transmitted_fraction());
    draw_time_scale(shell, ui);
    ImGui::Separator();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(shell.status_text().c_str());
    ImGui::PopTextWrapPos();
    ImGui::End();
}

template <typename ShellT>
void draw_landau_panel(ShellT& shell, UiState& ui, ses_shell::LandauApi& la) {
    ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse);
    draw_scene_picker(shell);
    draw_perf_readout(shell);
    if (ImGui::Button("Refire (2)")) shell.press('2');
    ImGui::SameLine();
    if (ImGui::Button("Ladder +B (3)")) shell.press('3');
    ImGui::SameLine();
    if (ImGui::Button("Ladder -B (4)")) shell.press('4');
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "a-dag / a: one cyclotron quantum (E_n = B(n + 1/2)).\n"
            "Up refuses past the lattice band ceiling; down refuses on the\n"
            "coherent orbit (a|alpha> = alpha|alpha>: no quantum to remove).");
    }
    ImGui::SameLine();
    if (ImGui::Button("Pause (Space)")) shell.toggle_pause();
    ImGui::SameLine();
    if (ImGui::Button("Face z (Z)")) shell.snap_camera_z();
    if (ImGui::SliderFloat("Field B (au)", &ui.la_b, 0.15f, 1.2f, "%.2f")) {
        la.set_field(static_cast<double>(ui.la_b));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Uniform B along z (exact plaquette flux).\n"
                          "omega_c = B; Landau levels E_n = B(n + 1/2).");
    }
    if (ImGui::SliderFloat("Launch k0 (au)", &ui.la_k0, 0.5f, 2.5f,
                           "%.2f")) {
        la.set_k0(static_cast<double>(ui.la_k0));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Cyclotron radius r = k0 / B: the amber circle "
                          "is the prediction,\nthe white trail is the "
                          "measured <r>(t).");
    }
    ImGui::Text("omega_c = %.2f, r = %.2f, <n> = %.1f", la.omega_c(),
                la.radius_pred(), la.mean_n());
    draw_time_scale(shell, ui);
    ImGui::Separator();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(shell.status_text().c_str());
    ImGui::PopTextWrapPos();
    ImGui::End();
}

template <typename ShellT>
void draw_bloch_panel(ShellT& shell, UiState& ui, ses_shell::BlochApi& bl) {
    ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse);
    draw_scene_picker(shell);
    draw_perf_readout(shell);
    if (ImGui::Button("Refire (2)")) shell.press('2');
    ImGui::SameLine();
    if (ImGui::Button("Pause (Space)")) shell.toggle_pause();
    ImGui::SameLine();
    if (ImGui::Button("Face z (Z)")) shell.snap_camera_z();
    if (ImGui::SliderFloat("Lattice depth V0", &ui.bl_v0, 0.0f, 4.0f,
                           "%.2f")) {
        bl.set_depth(static_cast<double>(ui.bl_v0));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("V0 sin^2(kL x) -- smooth (spectral accuracy; a "
                          "Kronig-Penney\nstep would Gibbs-ring in the FFT "
                          "basis). Deeper -> flatter bands,\nwider gaps "
                          "(first gap ~ V0/2 when weak).");
    }
    if (ImGui::SliderFloat("Tilt force F", &ui.bl_f, 0.0f, 0.15f,
                           "%.3f")) {
        bl.set_force(static_cast<double>(ui.bl_f));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Uniform force (comoving gauge, exact): q sweeps "
                          "the zone at rate F.\nInstead of accelerating "
                          "away the packet OSCILLATES -- Bloch\nperiod "
                          "T_B = 2 kL / F. Watch the cyan marker wrap the "
                          "inset.");
    }
    if (bl.force() > 0.0) {
        ImGui::Text("T_Bloch = %.0f au, q = %+.2f, max |dx| = %.1f",
                    bl.bloch_period(), bl.quasimomentum(), bl.excursion());
    } else {
        ImGui::Text("No tilt: band-limited dispersion only");
    }
    draw_time_scale(shell, ui);
    ImGui::Separator();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(shell.status_text().c_str());
    ImGui::PopTextWrapPos();
    ImGui::End();
}

template <typename ShellT>
void draw_corral_panel(ShellT& shell, UiState& ui, ses_shell::CorralApi& cr) {
    ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse);
    draw_scene_picker(shell);
    draw_perf_readout(shell);
    if (ImGui::Button("Ground (2)")) shell.press('2');
    ImGui::SameLine();
    if (ImGui::Button("Next state (3)")) shell.press('3');
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Relax the next standing-wave mode (deflated "
                          "against the captured\nones): the higher ripple "
                          "patterns of the STM image.");
    }
    ImGui::SameLine();
    if (ImGui::Button("Fermi wave (5)")) shell.press('5');
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "The state the STM actually images: the standing wave AT the\n"
            "Fermi energy (k_F R ~ j0_10: ~10 radial nodes), not the ground.");
    }
    ImGui::SameLine();
    if (ImGui::Button("Packet (F)")) shell.press('F');
    ImGui::SameLine();
    if (ImGui::Button("Pause (Space)")) shell.toggle_pause();
    ImGui::SameLine();
    if (ImGui::Button("Face z (Z)")) shell.snap_camera_z();
    ImGui::SliderFloat("Ring radius (Bohr)", &ui.cr_r, 6.0f, 12.0f, "%.1f");
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        cr.set_radius(static_cast<double>(ui.cr_r));
    }
    ImGui::Text("States %d%s, P(inside) = %.2f", cr.captured(),
                cr.relaxing() ? " (relaxing...)" : "", cr.confinement());
    draw_time_scale(shell, ui);
    ImGui::Separator();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(shell.status_text().c_str());
    ImGui::PopTextWrapPos();
    ImGui::End();
}

template <typename ShellT>
void draw_spin_panel(ShellT& shell, UiState& ui, ses_shell::SpinApi& sp) {
    ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse);
    draw_scene_picker(shell);
    draw_perf_readout(shell);
    if (ImGui::Button("Reset (2)")) shell.press('2');
    ImGui::SameLine();
    if (ImGui::Button("pi/2 (3)")) shell.press('3');
    ImGui::SameLine();
    if (ImGui::Button("pi (4)")) shell.press('4');
    ImGui::SameLine();
    if (ImGui::Button("Spin echo (5)")) shell.press('5');
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("64 detuned spins: pi/2, fan out for tau, pi,\n"
                          "refocus -- the mean arrow collapses and comes\n"
                          "BACK (Hahn echo).");
    }
    if (ImGui::Button(sp.rf_on() ? "RF off (L)" : "RF drive (L)")) {
        shell.press('L');
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Co-rotating circular drive locked to omega_L at\n"
                          "toggle time: resonant Rabi nutation (NMR).");
    }
    ImGui::SameLine();
    if (ImGui::Button("Measure (M)")) shell.measure_now();
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Born projection along B_hat: the arrow snaps\n"
                          "with |<+B|psi>|^2 to aligned, else anti.");
    }
    ImGui::SameLine();
    if (ImGui::Button("Pause (Space)")) shell.toggle_pause();
    const char* axes[3] = {"Bx", "By", "Bz"};
    for (int a = 0; a < 3; ++a) {
        float v = static_cast<float>(sp.b(a));
        if (ImGui::SliderFloat(axes[a], &v, -1.0f, 1.0f, "%.2f")) {
            sp.set_b(a, static_cast<double>(v));
        }
    }
    const char* eaxes[3] = {"Ex", "Ey", "Ez"};
    for (int a = 0; a < 3; ++a) {
        float v = static_cast<float>(sp.e(a));
        if (ImGui::SliderFloat(eaxes[a], &v, -1.0f, 1.0f, "%.2f")) {
            sp.set_e(a, static_cast<double>(v));
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("A pinned spin has no electric dipole: E only\n"
                          "rides as the red flux (plus a global phase).");
    }
    ImGui::Text("<s> = (%+.2f, %+.2f, %+.2f)   echo peak %.2f",
                sp.bloch_x(), sp.bloch_y(), sp.bloch_z(), sp.echo_peak());
    draw_time_scale(shell, ui);
    ImGui::Separator();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(shell.status_text().c_str());
    ImGui::PopTextWrapPos();
    ImGui::End();
}

template <typename ShellT>
void draw_spins_panel(ShellT& shell, UiState& ui, ses_shell::SpinsApi& sn) {
    ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse);
    draw_scene_picker(shell);
    draw_perf_readout(shell);
    if (ImGui::Button("Random (2)")) shell.press('2');
    ImGui::SameLine();
    if (ImGui::Button("Ferro seed (3)")) shell.press('3');
    ImGui::SameLine();
    if (ImGui::Button("Neel seed (4)")) shell.press('4');
    ImGui::SameLine();
    if (ImGui::Button("Measure (M)")) shell.measure_now();
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Born-project EVERY site onto +-B_hat.");
    }
    if (ImGui::Button(sn.exact_mode() ? "Mean-field (X)" : "Exact (X)")) {
        shell.press('X');
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Switch the engine:\n"
                          "Mean-field = product ansatz, arrows stay unit "
                          "length (no entanglement).\n"
                          "Exact = the full 2^16 wavefunction. Entanglement "
                          "is real, so the arrows SHRINK\n(mean |<s>| < 1) "
                          "as the sites correlate -- the thing mean-field "
                          "cannot show.");
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(sn.exact_mode() ? "[EXACT 2^16]"
                                           : "[mean-field]");
    float j = static_cast<float>(sn.j());
    if (ImGui::SliderFloat("Exchange J", &j, -1.0f, 1.0f, "%.2f")) {
        sn.set_j(static_cast<double>(j));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("J > 0: neighbors align (ferromagnet).\n"
                          "J < 0: they anti-align (Neel checkerboard).");
    }
    float al = static_cast<float>(sn.alpha());
    if (ImGui::SliderFloat("Damping alpha", &al, 0.0f, 0.3f, "%.2f")) {
        sn.set_alpha(static_cast<double>(al));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Gilbert damping: 0 = spin waves ripple "
                          "forever;\n> 0 = the lattice anneals into its "
                          "ordered ground.");
    }
    const char* axes[3] = {"Bx", "By", "Bz"};
    for (int a = 0; a < 3; ++a) {
        float v = static_cast<float>(sn.b(a));
        if (ImGui::SliderFloat(axes[a], &v, -1.0f, 1.0f, "%.2f")) {
            sn.set_b(a, static_cast<double>(v));
        }
    }
    ImGui::Text("|M| = %.2f   Neel = %.2f   mean |<s>| = %.2f",
                sn.magnetization(), sn.staggered(), sn.arrow_mean());
    if (sn.exact_mode() && ImGui::IsItemHovered()) {
        ImGui::SetTooltip("mean |<s>| < 1 = the sites are ENTANGLED "
                          "(exact only).");
    }
    draw_time_scale(shell, ui);
    ImGui::Separator();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(shell.status_text().c_str());
    ImGui::PopTextWrapPos();
    ImGui::End();
}

template <typename ShellT>
void draw_anderson_panel(ShellT& shell, UiState& ui,
                         ses_shell::AndersonApi& an) {
    ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse);
    draw_scene_picker(shell);
    draw_perf_readout(shell);
    if (ImGui::Button("Refire (2)")) an.refire();
    ImGui::SameLine();
    if (ImGui::Button("New landscape (5)")) an.reroll();
    ImGui::SameLine();
    if (ImGui::Button("Pause (Space)")) shell.toggle_pause();
    float w = static_cast<float>(an.disorder());
    if (ImGui::SliderFloat("Disorder W", &w, 0.0f, 2.0f, "%.2f")) {
        an.set_disorder(static_cast<double>(w));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Speckle strength. W = 0: the wire conducts\n"
                          "(ballistic flight). W ~ E: coherent multiple\n"
                          "scattering freezes the packet -- in 1D every\n"
                          "state is exponentially localized (Anderson).");
    }
    ImGui::Text("transmitted %.1f%%   on stage %.1f%%",
                100.0 * an.transmitted(), 100.0 * an.survived());
    draw_time_scale(shell, ui);
    ImGui::Separator();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(shell.status_text().c_str());
    ImGui::PopTextWrapPos();
    ImGui::End();
}

template <typename ShellT>
void draw_qpc_panel(ShellT& shell, UiState& ui, ses_shell::QpcApi& qp) {
    ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse);
    draw_scene_picker(shell);
    draw_perf_readout(shell);
    if (ImGui::Button("Fire (2)")) qp.fire();
    ImGui::SameLine();
    if (ImGui::Button("Pause (Space)")) shell.toggle_pause();
    float w = static_cast<float>(qp.gap());
    if (ImGui::SliderFloat("Gap width", &w, 1.0f, 10.0f, "%.1f")) {
        qp.set_gap(static_cast<double>(w));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("A transverse channel opens every lambda/2 =\n"
                          "pi/k0 ~ 3.1 Bohr of width: fire at several\n"
                          "widths and watch the transmitted flux climb\n"
                          "the conductance STAIRCASE.");
    }
    ImGui::Text("channels open: %d   transmitted: %.1f%%",
                qp.open_channels(), 100.0 * qp.transmitted());
    draw_time_scale(shell, ui);
    ImGui::Separator();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(shell.status_text().c_str());
    ImGui::PopTextWrapPos();
    ImGui::End();
}

template <typename ShellT>
void draw_rutherford_panel(ShellT& shell, UiState& ui,
                           ses_shell::RutherfordApi& rf) {
    ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse);
    draw_scene_picker(shell);
    draw_perf_readout(shell);
    if (ImGui::Button("Fire alpha (2)")) rf.refire();
    ImGui::SameLine();
    if (ImGui::Button("Pause (Space)")) shell.toggle_pause();
    ui.ru_e = static_cast<float>(rf.energy());
    if (ImGui::SliderFloat("Incident E (Ha)", &ui.ru_e, 5.0f, 80.0f, "%.1f")) {
        rf.set_energy(static_cast<double>(ui.ru_e));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Kinetic energy of the incoming packet. Lower E "
                          "turns around FARTHER out\n(r_min = 2Z/E) -- the "
                          "closest approach grows as the alpha slows.");
    }
    ui.ru_z = static_cast<float>(rf.z());
    if (ImGui::SliderFloat("Target Z (repulsion)", &ui.ru_z, 5.0f, 100.0f,
                           "%.0f")) {
        rf.set_z(static_cast<double>(ui.ru_z));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Nuclear charge of the target (default 79 = gold). "
                          "The repulsive\nCoulomb barrier V = +2Z/r; the "
                          "alpha projectile carries charge 2.");
    }
    ImGui::Text("r_min = 2Z/E = %.1f bohr   backscattered: %.0f%%",
                rf.turning_point(), 100.0 * rf.backscattered_fraction());
    draw_time_scale(shell, ui);
    ImGui::Separator();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(shell.status_text().c_str());
    ImGui::PopTextWrapPos();
    ImGui::End();
}

template <typename ShellT>
void draw_billiard_panel(ShellT& shell, UiState& ui,
                         ses_shell::BilliardApi& bl) {
    ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse);
    draw_scene_picker(shell);
    draw_perf_readout(shell);
    if (ImGui::Button("Fire packet (2)")) bl.fire();
    ImGui::SameLine();
    if (ImGui::Button(bl.stadium() ? "Circle (5)" : "Stadium (5)")) {
        bl.toggle_shape();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Circle: integrable -- |L| is conserved, the\n"
                          "orbit keeps a caustic hole. Stadium: chaotic --\n"
                          "the flat walls break L and the orbit fills the\n"
                          "table (Bunimovich).");
    }
    ImGui::SameLine();
    if (ImGui::Button(bl.avg_view() ? "Live view (A)" : "Average view (A)")) {
        bl.toggle_avg_view();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Time-averaged density: the caustic ring in the\n"
                          "circle, ergodic fill (with scars) in the stadium.");
    }
    ImGui::Text("center/interior avg: %.2f", bl.avg_center_fraction());
    ImGui::SameLine();
    if (ImGui::Button("Pause (Space)")) shell.toggle_pause();
    draw_time_scale(shell, ui);
    ImGui::Separator();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(shell.status_text().c_str());
    ImGui::PopTextWrapPos();
    ImGui::End();
}

template <typename ShellT>
void draw_qdot_panel(ShellT& shell, UiState& ui, ses_shell::QdotApi& qd) {
    ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse);
    draw_scene_picker(shell);
    draw_perf_readout(shell);
    if (ImGui::Button("Relax ground (2)")) shell.press('2');
    ImGui::SameLine();
    if (ImGui::Button("Displace (F)")) shell.press('F');
    ImGui::SameLine();
    if (ImGui::Button("Random packet (S)")) shell.press('S');
    if (ImGui::Button("Ladder +w (3)")) shell.press('3');
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Circular a_R-dag: +one w0 quantum, +1 L_z.\n"
                          "Needs B = 0 (the lattice gauge is not the\n"
                          "symmetric gauge the circular ladder lives in).");
    }
    ImGui::SameLine();
    if (ImGui::Button("Ladder -w (4)")) shell.press('4');
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Shift the relaxed ground sideways: a coherent "
                          "state. At B = 0 it\nswings at w0; with B it "
                          "traces the two-frequency Fock-Darwin\nrosette "
                          "(omega_pm = Omega -+ B/2), breadcrumbed white.");
    }
    ImGui::SameLine();
    if (ImGui::Button("Pause (Space)")) shell.toggle_pause();
    ImGui::SameLine();
    if (ImGui::Button("Face z (Z)")) shell.snap_camera_z();
    if (ImGui::SliderFloat("Dot stiffness w0", &ui.qd_w0, 0.2f, 1.0f,
                           "%.2f")) {
        qd.set_omega0(static_cast<double>(ui.qd_w0));
    }
    if (ImGui::SliderFloat("Field B", &ui.qd_b, 0.0f, 1.2f, "%.2f")) {
        qd.set_field(static_cast<double>(ui.qd_b));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Uniform B along z: the ground energy must land "
                          "at the\nFock-Darwin Omega = sqrt(w0^2 + B^2/4).");
    }
    ImGui::Text("E = %.4f vs Omega = %.4f%s", qd.energy_meas(),
                qd.energy_pred(), qd.relaxing() ? " (relaxing...)" : "");
    draw_ho_spectrum_strip(qd);
    draw_time_scale(shell, ui);
    ImGui::Separator();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(shell.status_text().c_str());
    ImGui::PopTextWrapPos();
    ImGui::End();
}

}  // namespace app
