module;
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
export module ses.scenario.selftest_arcs;
export import ses.scenario.manifold_spec;


// Verification + selftest arcs: every --dump-frame* and --selftest-* arc,
// registered against the live shell. Templated on the shell type so the
// shell class stays private to main.cpp. Each arc waits for the startup
// atlas build (run_when_manifold_ready) and then CHAINS scheduler timers, so
// a slower GPU stretches the run instead of false-failing a wall-clock
// verdict. The shell provides: sched() (Scheduler), request_exit(code),
// has_arg(name), dump_frame_bmp(path), and the control/probe wrappers.


export namespace ses_shell {

// Selftest helper: poll until the startup atlas build has produced the
// channel table, then run the arc (slower GPUs just stretch the wait).
template <typename ShellT, typename F>
void run_when_manifold_ready(ShellT* shell, F fn) {
    if (shell->manifold_ready()) {
        fn();
        return;
    }
    shell->sched().after(500,
                         [shell, fn] { run_when_manifold_ready(shell, fn); });
}

// --selftest-scene helper: poll (500 ms) until the CURRENT scene's sim time
// has advanced >= 0.2 au from its value at the first poll after the switch
// settles; ~30 s of no progress is the failure verdict. Logs the observed
// rate so a slow scene is diagnosable from the transcript.
template <typename ShellT, typename Done>
void selftest_scene_wait_running(ShellT* shell, const char* name, int polls,
                                 Done done, double t_start = -1.0) {
    constexpr int kMaxPolls = 60;
    const double t = shell->director().sim_time();
    if (t_start >= 0.0 && t - t_start >= 0.2) {
        std::fprintf(stderr, "selftest-scene: %s advanced %.2f au in %d polls\n",
                     name, t - t_start, polls);
        done(true);
        return;
    }
    if (polls >= kMaxPolls) {
        std::fprintf(stderr, "selftest-scene: %s STALLED at %.3f au\n", name, t);
        done(false);
        return;
    }
    const double anchor = t_start >= 0.0 ? t_start : t;
    shell->sched().after(500, [shell, name, polls, done, anchor] {
        selftest_scene_wait_running(shell, name, polls + 1, done, anchor);
    });
}

template <typename ShellT>
void register_verification_arcs(ShellT* shell) {
    // Render verification: read the finished scene image back to
    // frame_dump.bmp and exit; the readback verifies the whole ses_vk path
    // end to end. BMP is hand-rolled (no image library anywhere).
    if (shell->has_arg("--dump-frame")) {
        run_when_manifold_ready(shell, [shell] {
            shell->sched().after(2000, [shell] {
                const bool ok = shell->dump_frame_bmp("frame_dump.bmp");
                std::fprintf(stderr, "dump-frame: %ux%u  [%s]\n",
                             shell->frame_width(), shell->frame_height(),
                             ok ? "PASS" : "FAIL");
                shell->request_exit(ok ? 0 : 1);
            });
        });
    }
    // Same, from INSIDE the box: the volume pass rasterizes the proxy's back
    // faces (front-face culled), which only an interior eye exercises.
    if (shell->has_arg("--dump-frame-near")) {
        run_when_manifold_ready(shell, [shell] {
            std::fprintf(stderr, "dump-frame-near: manifold ready, zooming in\n");
            shell->debug_set_camera_distance(4.0);
            shell->sched().after(2000, [shell] {
                std::fprintf(stderr, "dump-frame-near: grabbing\n");
                const bool ok = shell->dump_frame_bmp("frame_dump_near.bmp");
                std::fprintf(stderr, "dump-frame-near: %ux%u  [%s]\n",
                             shell->frame_width(), shell->frame_height(),
                             ok ? "PASS" : "FAIL");
                shell->request_exit(ok ? 0 : 1);
            });
        });
    }

    // Surface-mode render verification: toggle to the GPU-extracted
    // isosurface and dump the finished frame -- mc_prepare/extract + the
    // indirect draw, end to end.
    if (shell->has_arg("--dump-frame-surface")) {
        run_when_manifold_ready(shell, [shell] {
            shell->toggle_view_mode();
            shell->sched().after(2000, [shell] {
                const bool ok = shell->dump_frame_bmp("frame_dump_surface.bmp");
                std::fprintf(stderr, "dump-frame-surface: %ux%u  [%s]\n",
                             shell->frame_width(), shell->frame_height(),
                             ok ? "PASS" : "FAIL");
                shell->request_exit(ok ? 0 : 1);
            });
        });
    }

    // Cross-section render verification: enable the clip plane AND the slice
    // sheet (z-normal, through the nucleus) and dump the frame -- exercises
    // both the volume clip path and the slice pipeline end to end.
    if (shell->has_arg("--dump-frame-slice")) {
        run_when_manifold_ready(shell, [shell] {
            shell->hy()->debug_prepare_state(k3DZ0);  // a lobed orbital (3d_z2)
            shell->enable_cross_section_demo();
            shell->sched().after(2500, [shell] {
                const bool ok = shell->dump_frame_bmp("frame_dump_slice.bmp");
                std::fprintf(stderr, "dump-frame-slice: %ux%u  [%s]\n",
                             shell->frame_width(), shell->frame_height(),
                             ok ? "PASS" : "FAIL");
                shell->request_exit(ok ? 0 : 1);
            });
        });
    }

    // Live scene switching (the panel combo's path): hydrogen -> harmonic ->
    // tunnel through the shell's deferred device-idle swap. Each hop must
    // produce a RUNNING scene: poll until sim time has advanced >= 0.2 au
    // (5 steps), chained not wall-clocked (repo lesson: fixed deadlines
    // false-fail on slower GPUs). Registered without the manifold wait on
    // purpose: switching away MID-atlas-build is part of the contract.
    if (shell->has_arg("--selftest-scene")) {
        shell->sched().after(2000, [shell] {
            const bool hy_ok = shell->hy() != nullptr;
            shell->request_scene(1);  // harmonic trap
            selftest_scene_wait_running(shell, "harmonic", 0, [shell, hy_ok](
                                                                 bool harm_runs) {
                const bool harm_ok =
                    shell->hy() == nullptr && shell->tn() == nullptr;
                // Scene-generic Larmor readout: the oscillating coherent
                // state must report a radiated power in the title.
                const bool harm_emits =
                    shell->status_text().find("emit P") != std::string::npos;
                std::fprintf(stderr, "selftest-scene: harmonic status: %s\n",
                             shell->status_text().c_str());
                shell->request_scene(2);  // tunneling barrier
                selftest_scene_wait_running(
                    shell, "tunnel", 0,
                    [shell, hy_ok, harm_ok, harm_emits,
                     harm_runs](bool tn_runs) {
                        const bool tn_ok = shell->tn() != nullptr;
                        const bool leg3d = hy_ok && harm_ok && harm_emits &&
                                           harm_runs && tn_ok && tn_runs;
                        std::fprintf(
                            stderr,
                            "selftest-scene: hydrogen %d, harmonic %d "
                            "(runs %d, emits %d), tunnel %d (runs %d)\n",
                            hy_ok, harm_ok, harm_runs, harm_emits, tn_ok,
                            tn_runs);
                        // The 1D legs: GPU scene -> CPU-only scene (renderer
                        // rebuild + overlay path), CPU -> CPU, and back to a
                        // GPU scene -- the reinit-safety gauntlet.
                        shell->request_scene(3);  // 1D harmonic ladder
                        selftest_scene_wait_running(shell, "harmonic1d", 0,
                                                    [shell, leg3d](
                                                        bool h1_runs) {
                            const bool h1_ok = shell->ln() != nullptr &&
                                               shell->hy() == nullptr;
                            shell->request_scene(4);  // 1D tunneling
                            selftest_scene_wait_running(shell, "tunnel1d", 0,
                                                        [shell, leg3d, h1_ok,
                                                         h1_runs](
                                                            bool t1_runs) {
                                const bool t1_ok = shell->tn() != nullptr &&
                                                   shell->ln() == nullptr;
                                shell->request_scene(1);  // back to a GPU scene
                                selftest_scene_wait_running(
                                    shell, "harmonic-return", 0,
                                    [shell, leg3d, h1_ok, h1_runs, t1_ok,
                                     t1_runs](bool ret_runs) {
                                        const bool pass = leg3d && h1_ok &&
                                                          h1_runs && t1_ok &&
                                                          t1_runs && ret_runs;
                                        std::fprintf(
                                            stderr,
                                            "selftest-scene: harmonic1d %d "
                                            "(runs %d), tunnel1d %d (runs %d), "
                                            "gpu-return runs %d  [%s]\n",
                                            h1_ok, h1_runs, t1_ok, t1_runs,
                                            ret_runs, pass ? "PASS" : "FAIL");
                                        shell->request_exit(pass ? 0 : 1);
                                    });
                            });
                        });
                    });
            });
        });
    }

    // Renderer survival across a scene switch: swap to the harmonic trap
    // (different grid -> full SceneRenderer rebuild) and dump the finished
    // frame -- the windowed combo's renderer path, verified headless.
    // CONDITION-POLLED, not wall-clocked: the swap defers the new scene's
    // compute init to the loop's init block, which runs AFTER the scheduler
    // poll -- a fixed-delay dump can fire before the first post-swap render
    // and read a never-rendered target (latent race, found the hard way).
    if (shell->has_arg("--dump-frame-switch")) {
        shell->sched().after(2000, [shell] {
            shell->request_scene(1);  // harmonic trap
            selftest_scene_wait_running(shell, "switch-target", 0, [shell](
                                                                      bool runs) {
                if (!runs) {
                    std::fprintf(stderr, "dump-frame-switch: scene STALLED  "
                                         "[FAIL]\n");
                    shell->request_exit(1);
                    return;
                }
                shell->sched().after(1000, [shell] {  // let a few frames settle
                    const bool ok =
                        shell->dump_frame_bmp("frame_dump_switch.bmp");
                    std::fprintf(stderr, "dump-frame-switch: %ux%u  [%s]\n",
                                 shell->frame_width(), shell->frame_height(),
                                 ok ? "PASS" : "FAIL");
                    shell->request_exit(ok ? 0 : 1);
                });
            });
        });
    }

    // Trap Fock-ladder decay: prepare an eigenstate (Key 5 -> 1p_z), arm the
    // Einstein-A jumps (Key D), and expect >= 1 photon inside the window --
    // the QED half of the trap's complementarity demo (the license physics
    // lives in tests/trap_ladder_test.cpp). tau_display ~ 8 au as the atom.
    if (shell->has_arg("--selftest-trapdecay")) {
        shell->sched().after(2000, [shell] {
            shell->press('5');  // 1p_z: N = 1, one rung above ground
            shell->sched().after(1000, [shell] {
                const long long baseline = shell->director().photon_count();
                shell->press('D');  // arm: the window starts now
                shell->sched().after(30000, [shell, baseline] {
                    const long long fresh =
                        shell->director().photon_count() - baseline;
                    std::fprintf(stderr,
                                 "selftest-trapdecay: photons = %lld  [%s]\n",
                                 fresh, fresh >= 1 ? "PASS" : "FAIL");
                    shell->request_exit(fresh >= 1 ? 0 : 1);
                });
            });
        });
    }

    // Decay arc: prepare 2p WITH DECAY OFF (relaxation auto-completes into
    // real time, and an armed 2p would fire its one photon BEFORE the
    // baseline -- the n<=6 seed converges fast enough to lose that race
    // most runs), then arm decay exactly when the counting window opens.
    // >= 1 jump expected: tau_display ~ 8 au vs a ~75 au window.
    if (shell->has_arg("--selftest-decay")) {
        run_when_manifold_ready(shell, [shell] {
            shell->hy()->toggle_decay();      // OFF: hold the prepared 2p
            shell->hy()->relax_to_excited();  // caches ready: no block
            shell->sched().after(13500, [shell] {
                const long long baseline = shell->hy()->photon_count();
                shell->set_real_time();
                shell->hy()->toggle_decay();  // ON: the window starts NOW
                shell->sched().after(30000, [shell, baseline] {
                    const long long fresh = shell->hy()->photon_count() - baseline;
                    std::fprintf(stderr, "selftest-decay: photons = %lld  [%s]\n",
                                 fresh, fresh >= 1 ? "PASS" : "FAIL");
                    shell->request_exit(fresh >= 1 ? 0 : 1);
                });
            });
        });
    }

    // Energy-measurement arc: relax to 1s (decay OFF so the state stays
    // put); a projective energy measurement must report the 1s eigenstate.
    if (shell->has_arg("--selftest-energy")) {
        run_when_manifold_ready(shell, [shell] {
            shell->hy()->toggle_decay();  // OFF: keep the relaxed state stationary
            shell->hy()->set_relaxing();  // cool to 1s
            // 20 s: the random n <= 6 seed carries only ~1% weight in 1s, so
            // the ITP descent to ground needs a generous window to converge.
            shell->sched().after(20000, [shell] {
                shell->hy()->measure_energy_now();
                shell->sched().after(1500, [shell] {
                    const int idx = shell->hy()->last_measured_index();
                    const bool pass = idx == kS1;
                    std::fprintf(
                        stderr, "selftest-energy: measured %s  [%s]\n",
                        idx >= 0 ? kStateSpec[static_cast<std::size_t>(idx)].name
                                 : "outside-manifold",
                        pass ? "PASS" : "FAIL");
                    shell->request_exit(pass ? 0 : 1);
                });
            });
        });
    }

    // Static-E arc: relax to 1s (<z> ~ 0), switch on a sub-ionization +z
    // field, require <z> to shift measurably (Stark polarization).
    if (shell->has_arg("--selftest-efield")) {
        run_when_manifold_ready(shell, [shell] {
            shell->hy()->toggle_decay();  // OFF: keep the state put
            shell->hy()->set_relaxing();  // cool to 1s
            shell->sched().after(20000, [shell] {
                const double z0 = shell->hy()->mean_z();
                shell->set_real_time();
                shell->hy()->set_efield_e0(0.02);  // sub-ionization: clean polarization
                shell->sched().after(15000, [shell, z0] {
                    const double z1 = shell->hy()->mean_z();
                    const bool pass = std::abs(z1 - z0) > 0.03;
                    std::fprintf(stderr,
                                 "selftest-efield: <z> %.4f -> %.4f Bohr  [%s]\n",
                                 z0, z1, pass ? "PASS" : "FAIL");
                    shell->request_exit(pass ? 0 : 1);
                });
            });
        });
    }

    // Pump arc: relax to 1s, laser ON (Z), require peak P(2pz) >= 0.5; then
    // decay ON and require >= 2 photons (a ground-start run without the
    // pump emits zero, so 2 is unambiguous).
    if (shell->has_arg("--selftest-rabi")) {
        run_when_manifold_ready(shell, [shell] {
            shell->hy()->toggle_decay();  // OFF: study the clean coherent flop
            shell->hy()->set_relaxing();  // cool to 1s
            shell->sched().after(20000, [shell] {
                shell->set_real_time();
                shell->hy()->toggle_laser();  // cached: instant
                // 256^3 runs ~3 au/s of sim time: the half-flop (pi/Omega
                // ~ 79 au) needs most of this window.
                shell->sched().after(60000, [shell] {
                    const double peak = shell->hy()->peak_excited_population();
                    std::fprintf(stderr, "selftest-rabi: peak P(2pz) = %.3f  [%s]\n",
                                 peak, peak >= 0.5 ? "PASS" : "FAIL");
                    if (peak < 0.5) {
                        shell->request_exit(1);
                        return;
                    }
                    const long long baseline = shell->hy()->photon_count();
                    shell->hy()->toggle_decay();  // back ON: fluorescence
                    shell->sched().after(180000, [shell, baseline] {
                        const long long fresh =
                            shell->hy()->photon_count() - baseline;
                        std::fprintf(stderr,
                                     "selftest-rabi: photons = %lld  [%s]\n",
                                     fresh, fresh >= 2 ? "PASS" : "FAIL");
                        shell->request_exit(fresh >= 2 ? 0 : 1);
                    });
                });
            });
        });
    }

    // Cascade arc: excite 3d_z2 and require >= 2 photons -- 3d cannot reach
    // 1s directly (dl = 2), so two photons prove 3d -> 2p -> 1s fired.
    // EARLY-EXITS the moment 2 photons land (the count only grows); the 120 s
    // window is just the worst-case FAIL bound, since step throughput is GPU-
    // and policy-dependent. time_scale is maxed (16, the clamp ceiling) for
    // the widest lifetime margin on the slowest GPU.
    if (shell->has_arg("--selftest-cascade")) {
        run_when_manifold_ready(shell, [shell] {
            const long long baseline = shell->hy()->photon_count();
            shell->hy()->excite_n3();  // first in the cycle: 3d_z2
            shell->set_time_scale(16);
            auto probe = std::make_shared<int>(-1);
            *probe = shell->sched().every(1000, [shell, baseline, probe] {
                if (shell->hy()->photon_count() - baseline >= 2) {
                    shell->sched().cancel(*probe);
                    std::fprintf(stderr, "selftest-cascade: photons = %lld  [PASS]\n",
                                 shell->hy()->photon_count() - baseline);
                    shell->request_exit(0);
                }
            });
            shell->sched().after(120000, [shell, baseline, probe] {
                shell->sched().cancel(*probe);
                const long long fresh = shell->hy()->photon_count() - baseline;
                std::fprintf(stderr, "selftest-cascade: photons = %lld  [%s]\n",
                             fresh, fresh >= 2 ? "PASS" : "FAIL");
                shell->request_exit(fresh >= 2 ? 0 : 1);
            });
        });
    }

    // Magnetic arc: prepare 2p_x (decay off), B along z, require P(2p_y) to
    // rise past 0.3 -- proving psi ITSELF precesses. Probed periodically so
    // the sin^2 oscillation phase cannot alias the verdict.
    if (shell->has_arg("--selftest-magnetic")) {
        run_when_manifold_ready(shell, [shell] {
            shell->hy()->toggle_decay();            // decay OFF: pure precession
            shell->hy()->debug_prepare_state(kP2X);  // psi = 2p_x
            // Modest field: a large B would diamagnetically deform the state
            // and cap the overlap. Magnetic stepping is slow (~0.4 au/s), so
            // the window is long.
            shell->hy()->set_bfield_b(0.08);         // B along z (default axis)
            auto max_py = std::make_shared<double>(0.0);
            const int probe = shell->sched().every(1500, [shell, max_py] {
                *max_py = std::max(*max_py, shell->hy()->probe_population(kP2Y));
            });
            shell->sched().after(90000, [shell, max_py, probe] {
                shell->sched().cancel(probe);
                std::fprintf(stderr, "selftest-magnetic: max P(2p_y) = %.3f  [%s]\n",
                             *max_py, *max_py > 0.3 ? "PASS" : "FAIL");
                shell->request_exit(*max_py > 0.3 ? 0 : 1);
            });
        });
    }

    // Manifold arc: deterministic channel-table checks (selection rule, 2p
    // degeneracy, ordering, cascade, dl rule), then an X-pol pump from 1s --
    // its photons can only flow through 2p_x, proving non-2p_z channels fire.
    if (shell->has_arg("--selftest-manifold")) {
        run_when_manifold_ready(shell, [shell] {
            const double a_pz = shell->hy()->channel_a(kP2Z, kS1);
            const double a_px = shell->hy()->channel_a(kP2X, kS1);
            const double a_2s1s = shell->hy()->channel_a(kS2, kS1);
            std::fprintf(stderr,
                         "selftest-manifold: A(2pz->1s)=%.3e A(2px->1s)=%.3e "
                         "A(2s->1s)=%.3e  E(1s)=%.4f E(2pz)=%.4f E(2s)=%.4f\n",
                         a_pz, a_px, a_2s1s, shell->hy()->state_energy(kS1),
                         shell->hy()->state_energy(kP2Z), shell->hy()->state_energy(kS2));
            const bool selection = a_pz > 0.0 && a_2s1s < 1e-3 * a_pz;
            const bool degeneracy = a_pz > 0.0 && std::abs(a_px / a_pz - 1.0) < 0.05;
            const bool ordering =
                shell->hy()->state_energy(kS1) < shell->hy()->state_energy(kP2Z) &&
                shell->hy()->state_energy(kS1) < shell->hy()->state_energy(kS2);
            // The n = 3 shell: cascade paths open, Delta-l selection
            // rules hold (3s -> 1s and 3d -> 1s are E1-forbidden).
            const double a_3s2p = shell->hy()->channel_a(k3S, kP2Z);
            const double a_3d2p = shell->hy()->channel_a(k3DZ0, kP2Z);
            const double a_3s1s = shell->hy()->channel_a(k3S, kS1);
            const double a_3d1s = shell->hy()->channel_a(k3DZ0, kS1);
            const bool cascade = a_3s2p > 0.0 && a_3d2p > 0.0;
            const bool dl_rule =
                a_3d2p > 0.0 && a_3s1s < 1e-3 * a_3d2p && a_3d1s < 1e-3 * a_3d2p;
            std::fprintf(stderr,
                         "selftest-manifold: A(3s->2pz)=%.3e A(3dz2->2pz)=%.3e "
                         "A(3s->1s)=%.3e A(3dz2->1s)=%.3e\n",
                         a_3s2p, a_3d2p, a_3s1s, a_3d1s);
            std::fprintf(stderr,
                         "selftest-manifold: selection %s, degeneracy %s, ordering "
                         "%s, cascade %s, dl-rule %s\n",
                         selection ? "PASS" : "FAIL", degeneracy ? "PASS" : "FAIL",
                         ordering ? "PASS" : "FAIL", cascade ? "PASS" : "FAIL",
                         dl_rule ? "PASS" : "FAIL");
            if (!(selection && degeneracy && ordering && cascade && dl_rule)) {
                shell->request_exit(1);
                return;
            }
            shell->hy()->set_relaxing();  // cool to 1s for the X-pol pump
            shell->sched().after(20000, [shell] {
                shell->set_real_time();
                shell->hy()->toggle_laser();  // Z (cached: no block)
                shell->hy()->toggle_laser();  // -> X
                const long long baseline = shell->hy()->photon_count();
                // Two X-pol photons need ~170 au of sim time (~2 half-flops
                // + accelerated lifetimes); 180 s gives real margin at the
                // slowest throughput.
                shell->sched().after(180000, [shell, baseline] {
                    const long long fresh = shell->hy()->photon_count() - baseline;
                    std::fprintf(stderr,
                                 "selftest-manifold: x-pol photons = %lld  [%s]\n",
                                 fresh, fresh >= 2 ? "PASS" : "FAIL");
                    shell->request_exit(fresh >= 2 ? 0 : 1);
                });
            });
        });
    }

    // Partial-measurement arc: (2s+2p_z)/sqrt(2) is a DEGENERATE shell
    // superposition -- a true n-shell measurement must return n=2 with BOTH
    // populations intact (intra-shell coherence survives); a following l
    // measurement picks one l and must leave a pure subspace. Then L_z on a
    // prepared p_x: outcome +-1 with equal cos/sin populations (a ring),
    // and an immediate repeat must agree (projective repeatability).
    if (shell->has_arg("--selftest-partial")) {
        run_when_manifold_ready(shell, [shell] {
            shell->hy()->toggle_decay();  // OFF: jumps would race the assertions
            shell->hy()->debug_prepare_superposition(kS2, kP2Z);
            shell->hy()->measure_n_shell_now();
            shell->sched().after(800, [shell] {
                const double ps = shell->hy()->probe_population(kS2);
                const double pz = shell->hy()->probe_population(kP2Z);
                const bool n_ok = shell->hy()->last_partial_outcome() == 2 &&
                                  ps > 0.35 && ps < 0.65 && pz > 0.35 &&
                                  pz < 0.65;
                std::fprintf(stderr,
                             "selftest-partial: n-shell -> %d, P(2s)=%.2f "
                             "P(2pz)=%.2f  [%s]\n",
                             shell->hy()->last_partial_outcome(), ps, pz,
                             n_ok ? "PASS" : "FAIL");
                shell->hy()->measure_l_now();
                shell->sched().after(800, [shell, n_ok] {
                    const int l = shell->hy()->last_partial_outcome();
                    const double ps = shell->hy()->probe_population(kS2);
                    const double pz = shell->hy()->probe_population(kP2Z);
                    const bool l_ok = (l == 0 && ps > 0.98) ||
                                      (l == 1 && pz > 0.98);
                    std::fprintf(stderr,
                                 "selftest-partial: l -> %d, P(2s)=%.2f "
                                 "P(2pz)=%.2f  [%s]\n",
                                 l, ps, pz, l_ok ? "PASS" : "FAIL");
                    shell->hy()->debug_prepare_state(kP2X);
                    shell->hy()->measure_m_now();
                    shell->sched().after(800, [shell, n_ok, l_ok] {
                        const int m1 = shell->hy()->last_partial_outcome();
                        const double px = shell->hy()->probe_population(kP2X);
                        const double py = shell->hy()->probe_population(kP2Y);
                        const bool ring_ok =
                            (m1 == 1 || m1 == -1) && px > 0.35 && px < 0.65 &&
                            py > 0.35 && py < 0.65;
                        std::fprintf(stderr,
                                     "selftest-partial: m -> %+d, "
                                     "P(px)=%.2f P(py)=%.2f  [%s]\n",
                                     m1, px, py, ring_ok ? "PASS" : "FAIL");
                        shell->hy()->measure_m_now();
                        shell->sched().after(800, [shell, n_ok, l_ok, ring_ok,
                                                   m1] {
                            const bool rep_ok =
                                shell->hy()->last_partial_outcome() == m1;
                            const bool pass =
                                n_ok && l_ok && ring_ok && rep_ok;
                            std::fprintf(
                                stderr,
                                "selftest-partial: repeat m -> %+d "
                                "(repeatability %s)  [%s]\n",
                                shell->hy()->last_partial_outcome(),
                                rep_ok ? "PASS" : "FAIL",
                                pass ? "PASS" : "FAIL");
                            shell->request_exit(pass ? 0 : 1);
                        });
                    });
                });
            });
        });
    }

    // 1D ladder arc (main forces --scene=harmonic1d): the Fock chain through
    // the Ladder1dApi seam. Two raises must land n = 2 with <H> = 2.5 w; two
    // lowers return to ground; the third lower must be REFUSED by the
    // operator itself (a|0> = 0) with the level untouched. Condition-polled
    // (the CPU scene ticks immediately, but the poll keeps the boot-order
    // contract explicit).
    if (shell->has_arg("--selftest-ladder1d")) {
        shell->sched().after(1000, [shell] {
            selftest_scene_wait_running(shell, "harmonic1d", 0, [shell](
                                                                    bool runs) {
                auto* ln = shell->ln();
                if (!runs || ln == nullptr) {
                    std::fprintf(stderr, "selftest-ladder1d: scene not "
                                         "running or no api  [FAIL]\n");
                    shell->request_exit(1);
                    return;
                }
                const bool up_ok = ln->ladder(true) && ln->ladder(true) &&
                                   ln->level() == 2;
                const double e2 = ln->level_energy();
                const bool e_ok = std::abs(e2 - 2.5 * 0.25) < 1e-3;
                const bool down_ok = ln->ladder(false) && ln->ladder(false) &&
                                     ln->level() == 0;
                const bool refuse_ok = !ln->ladder(false) && ln->level() == 0;
                // Superposition: Var(H) must classify it (level -1), and
                // the ladder must act on it linearly (no refusal -- every
                // component is reachable).
                ln->random_superposition();
                const bool mix_ok = ln->level() == -1 && ln->ladder(true) &&
                                    ln->level() == -1;
                // Quench: stiffen the well under the kept psi, then reset
                // must land in the NEW well's ground at E = w/2 = 0.25 Ha.
                ln->set_omega(0.5);
                shell->reset_simulation();
                const bool quench_ok =
                    ln->omega() == 0.5 && ln->level() == 0 &&
                    std::abs(ln->level_energy() - 0.25) < 1e-3 &&
                    ln->max_level() > 10;  // stiffer well: higher clean cap
                // Stable-rung round trip: 20 up + 20 down at w = 0.5 --
                // far past the raw chain's cap (14), where the raw
                // operators disintegrate into high-k garbage on descent
                // (the observed ladder-down instability). Stable rungs
                // must land exactly back on the ground.
                bool trip_ok = ln->level() == 0;
                for (int i = 0; i < 20 && trip_ok; ++i) {
                    trip_ok = ln->ladder(true);
                }
                trip_ok = trip_ok && ln->level() == 20;
                for (int i = 0; i < 20 && trip_ok; ++i) {
                    trip_ok = ln->ladder(false);
                }
                trip_ok = trip_ok && ln->level() == 0 &&
                          std::abs(ln->level_energy() - 0.25) < 1e-3;
                // Measured cap peaks near w ~ 1 then FALLS: cranking the
                // well past the peak lowers the clean ladder cap (the whole
                // point of the empirical probe + widened slider). The
                // quenched state is a superposition, so these report the
                // raw-chain caps.
                ln->set_omega(1.0);
                const int cap_peak = ln->max_level();
                ln->set_omega(4.0);
                const int cap_high = ln->max_level();
                const bool peak_ok = cap_peak >= 14 && cap_peak > cap_high;
                const bool pass = up_ok && e_ok && down_ok && refuse_ok &&
                                  mix_ok && quench_ok && trip_ok && peak_ok;
                std::fprintf(stderr,
                             "selftest-ladder1d: up %d (E2 = %.4f Ha, ok %d), "
                             "down %d, ground-refuse %d, superposition %d, "
                             "quench %d, stable-trip20 %d, cap-peak %d "
                             "(w=1 cap %d > w=4 cap %d)  [%s]\n",
                             up_ok, e2, e_ok, down_ok, refuse_ok, mix_ok,
                             quench_ok, trip_ok, peak_ok, cap_peak, cap_high,
                             pass ? "PASS" : "FAIL");
                shell->request_exit(pass ? 0 : 1);
            });
        });
    }

    // 1D tunneling arc (main forces --scene=tunnel1d): same physics as the
    // 3D arc on its textbook axis. CPU steps are cheap, so time_scale 8
    // (~40 au/s) settles the transmitted lobe within seconds; assert a
    // classically-forbidden T in a sane band.
    if (shell->has_arg("--selftest-tunnel1d")) {
        shell->sched().after(1000, [shell] {
            selftest_scene_wait_running(shell, "tunnel1d", 0, [shell](
                                                                  bool runs) {
                if (!runs || shell->tn() == nullptr) {
                    std::fprintf(stderr, "selftest-tunnel1d: scene not "
                                         "running or no api  [FAIL]\n");
                    shell->request_exit(1);
                    return;
                }
                shell->set_time_scale(8);
                shell->sched().after(20000, [shell] {
                    const double t = shell->tn()->transmitted_max();
                    const bool pass = t > 1e-3 && t < 0.5;
                    std::fprintf(stderr,
                                 "selftest-tunnel1d: max T = %.4f  [%s]\n", t,
                                 pass ? "PASS" : "FAIL");
                    shell->request_exit(pass ? 0 : 1);
                });
            });
        });
    }

    // Tunneling arc (main forces --scene=tunnel): the packet launched at
    // x = -30 with v = 0.5 reaches the slab at ~60 au; the transmitted lobe
    // is fully past it well before ~150 au (~2.5 min at ~1 au/s). Assert a
    // classically-forbidden transmitted fraction in a sane band.
    if (shell->has_arg("--selftest-tunnel")) {
        run_when_manifold_ready(shell, [shell] {
            shell->sched().after(180000, [shell] {
                const double t = shell->tn()->transmitted_max();
                const bool pass = t > 1e-3 && t < 0.9;
                std::fprintf(stderr, "selftest-tunnel: max T = %.4f  [%s]\n", t,
                             pass ? "PASS" : "FAIL");
                shell->request_exit(pass ? 0 : 1);
            });
        });
    }
}

}  // namespace ses_shell
