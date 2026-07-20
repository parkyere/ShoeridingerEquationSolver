module;
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
export module ses.scenario.selftest_arcs;
export import ses.scenario.manifold_spec;
import ses.scenario.kepler_seed;


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

// Poll (1 s) until the CURRENT scene's sim time reaches t_target au; ~2 min
// without arrival is the stall verdict. Used where the physics needs a
// known span of SIMULATED time and the step cost is machine-dependent.
template <typename ShellT, typename Done>
void selftest_wait_sim_time(ShellT* shell, double t_target, int polls,
                            Done done) {
    if (shell->director().sim_time() >= t_target) {
        done(true);
        return;
    }
    if (polls >= 120) {
        done(false);
        return;
    }
    shell->sched().after(1000, [shell, t_target, polls, done] {
        selftest_wait_sim_time(shell, t_target, polls + 1, done);
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
                std::fprintf(stderr, "dump-frame: %ux%u (t = %.2f au)  [%s]\n",
                             shell->frame_width(), shell->frame_height(),
                             shell->director().sim_time(),
                             ok ? "PASS" : "FAIL");
                shell->request_exit(ok ? 0 : 1);
            });
        });
    }
    // Same, but AFTER real evolution: fast-forward to sim time 60 au at
    // time scale 16 first (the interference scenes need the transit done
    // before there is anything on the screen to look at).
    if (shell->has_arg("--dump-frame-late")) {
        run_when_manifold_ready(shell, [shell] {
            shell->set_time_scale(16);
            selftest_wait_sim_time(shell, 60.0, 0, [shell](bool ok_wait) {
                const bool ok =
                    ok_wait && shell->dump_frame_bmp("frame_dump_late.bmp");
                std::fprintf(stderr,
                             "dump-frame-late: %ux%u (t = %.2f au)  [%s]\n",
                             shell->frame_width(), shell->frame_height(),
                             shell->director().sim_time(),
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
            if (shell->hy() == nullptr) {  // arc needs the hydrogen manifold
                std::fprintf(stderr, "dump-frame-slice: requires the hydrogen "
                                     "scene (--scene=hydrogen)  [FAIL]\n");
                shell->request_exit(1);
                return;
            }
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
                    // The spectrometer must have recorded the 2p -> 1s
                    // photon at the Lyman-alpha energy 10.20 eV.
                    const int nl = shell->hy()->spectro_count();
                    const double ev =
                        nl > 0 ? shell->hy()->spectro_ev(nl - 1) : 0.0;
                    const bool line_ok =
                        fresh < 1 || std::abs(ev - 10.20) < 0.15;
                    const bool pass = fresh >= 1 && line_ok;
                    std::fprintf(stderr,
                                 "selftest-decay: photons = %lld, last line "
                                 "%.2f eV  [%s]\n",
                                 fresh, ev, pass ? "PASS" : "FAIL");
                    shell->request_exit(pass ? 0 : 1);
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
                shell->set_time_scale(16);  // explicit dial (no hidden pump boost)
                shell->hy()->toggle_laser();  // cached: instant
                // GPU-bound 256^3 lands ~3 au/s regardless of the dial: the
                // half-flop (pi/Omega ~ 79 au) needs most of this window.
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
    // Kepler packet arc: the circular-state superposition ORBITS the
    // nucleus CCW at the Kepler rate (correspondence principle). Polls
    // <x>,<y>, accumulates the unwrapped azimuth, and gates the measured
    // angular rate into a window around 1/n_bar^3 (the low-n tail and the
    // grid radial overlaps skew the weighted mean, hence the width).
    if (shell->has_arg("--selftest-kepler")) {
        run_when_manifold_ready(shell, [shell] {
            shell->hy()->toggle_decay();  // pure orbital beat
            shell->hy()->seed_kepler();
            struct Orbit {
                double phi_prev = 0.0;
                double phi_acc = 0.0;
                double t0 = -1.0;
                double r_lo = 1e9;
                double r_hi = 0.0;
                double z_hi = 0.0;
            };
            auto ob = std::make_shared<Orbit>();
            const double pi = 3.14159265358979323846;
            const int probe = shell->sched().every(1500, [shell, ob, pi] {
                const double x = shell->hy()->mean_x();
                const double y = shell->hy()->mean_y();
                const double phi = std::atan2(y, x);
                const double r = std::sqrt(x * x + y * y);
                ob->r_lo = std::min(ob->r_lo, r);
                ob->r_hi = std::max(ob->r_hi, r);
                ob->z_hi = std::max(ob->z_hi, std::abs(shell->hy()->mean_z()));
                if (ob->t0 < 0.0) {
                    ob->t0 = shell->director().sim_time();
                } else {
                    double d = phi - ob->phi_prev;
                    while (d > pi) d -= 2.0 * pi;
                    while (d < -pi) d += 2.0 * pi;
                    ob->phi_acc += d;
                }
                ob->phi_prev = phi;
            });
            shell->sched().after(90000, [shell, ob, probe] {
                shell->sched().cancel(probe);
                const double dt = shell->director().sim_time() - ob->t0;
                const double n_bar = kKeplerNBar;
                const double w_pred = 1.0 / (n_bar * n_bar * n_bar);
                const double rate = dt > 0.0 ? ob->phi_acc / dt : 0.0;
                const bool ccw = ob->phi_acc > 0.0;
                const bool rate_ok =
                    rate > 0.6 * w_pred && rate < 2.0 * w_pred;
                const bool radius_ok = ob->r_lo > 8.0 && ob->r_hi < 35.0;
                const bool planar_ok = ob->z_hi < 4.0;
                const bool pass =
                    dt > 40.0 && ccw && rate_ok && radius_ok && planar_ok;
                std::fprintf(stderr,
                             "selftest-kepler: dphi = %.2f rad over %.0f au "
                             "(rate %.4f vs 1/n^3 %.4f), r in [%.1f, %.1f], "
                             "|z| < %.2f  [%s]\n",
                             ob->phi_acc, dt, rate, w_pred, ob->r_lo,
                             ob->r_hi, ob->z_hi, pass ? "PASS" : "FAIL");
                shell->request_exit(pass ? 0 : 1);
            });
        });
    }

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
                shell->set_time_scale(16);  // explicit dial (no hidden pump boost)
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
    // Spin arc (main forces --scene=spin): Larmor quarter turn, resonant
    // Rabi inversion under the RF drive, then the Hahn echo refocus.
    // Grid-free scene -- the exact-rotation contracts live in spin_test.
    if (shell->has_arg("--selftest-spin")) {
        shell->sched().after(1000, [shell] {
            auto* sp = shell->sp();
            if (sp == nullptr) {
                std::fprintf(stderr, "selftest-spin: no api  [FAIL]\n");
                shell->request_exit(1);
                return;
            }
            shell->set_time_scale(16);
            const double t_quarter = 0.25 * 2.0 * 3.14159265358979323846 /
                                     0.5;  // omega_L = |B| = 0.5
            selftest_wait_sim_time(shell, t_quarter, 0, [shell](bool ok1) {
                const double y_quarter = shell->sp()->bloch_y();
                shell->sp()->toggle_rf();
                auto min_z = std::make_shared<double>(1.0);
                const int probe = shell->sched().every(120, [shell, min_z] {
                    *min_z = std::min(*min_z, shell->sp()->bloch_z());
                });
                // One FULL nutation (2 pi / Omega_R = 126 au): the sense
                // may swing +z first, and -1 only comes on the far side.
                const double t_mark = shell->director().sim_time();
                selftest_wait_sim_time(
                    shell, t_mark + 130.0, 0,
                    [shell, ok1, y_quarter, min_z, probe](bool ok2) {
                        shell->sched().cancel(probe);
                        shell->sp()->toggle_rf();
                        shell->sp()->spin_echo();
                        const double t2 = shell->director().sim_time();
                        selftest_wait_sim_time(
                            shell, t2 + 2.0 * 30.0 + 2.0, 0,
                            [shell, ok1, ok2, y_quarter,
                             min_z](bool ok3) {
                                const double echo =
                                    shell->sp()->echo_peak();
                                const bool larmor_ok = y_quarter > 0.8;
                                const bool rabi_ok = *min_z < -0.8;
                                const bool echo_ok = echo > 0.9;
                                const bool pass = ok1 && ok2 && ok3 &&
                                                  larmor_ok && rabi_ok &&
                                                  echo_ok;
                                std::fprintf(
                                    stderr,
                                    "selftest-spin: quarter-turn <y> "
                                    "%.2f, Rabi min<z> %.2f, echo %.2f  "
                                    "[%s]\n",
                                    y_quarter, *min_z, echo,
                                    pass ? "PASS" : "FAIL");
                                shell->request_exit(pass ? 0 : 1);
                            });
                    });
            });
        });
    }

    // Spin-lattice arc (main forces --scene=spins): damped J > 0 orders
    // a random boot into a ferromagnet; damped J < 0 into Neel. The
    // integrator contracts live in tests/spinlattice_test.cpp.
    if (shell->has_arg("--selftest-spins")) {
        shell->sched().after(1000, [shell] {
            auto* sn = shell->sn();
            if (sn == nullptr) {
                std::fprintf(stderr, "selftest-spins: no api  [FAIL]\n");
                shell->request_exit(1);
                return;
            }
            shell->set_time_scale(16);
            sn->set_alpha(0.1);
            sn->set_j(0.5);
            sn->seed_random();
            selftest_wait_sim_time(shell, 120.0, 0, [shell](bool ok1) {
                const double m_ferro = shell->sn()->magnetization();
                shell->sn()->set_j(-0.5);
                shell->sn()->seed_random();
                selftest_wait_sim_time(
                    shell, 240.0, 0, [shell, ok1, m_ferro](bool ok2) {
                        const double neel = shell->sn()->staggered();
                        const double m_res =
                            shell->sn()->magnetization();
                        // Third leg: switch to the EXACT 2^16 engine and
                        // let the Neel product entangle -- the mean arrow
                        // length must SHRINK below unity (the mean-field
                        // ansatz cannot do this).
                        shell->sn()->set_exact(true);
                        shell->sn()->set_alpha(0.0);
                        shell->sn()->set_b(2, 0.1);
                        shell->sn()->seed_neel();  // resets sim_time to 0
                        const double arrow0 = shell->sn()->arrow_mean();
                        selftest_wait_sim_time(
                            shell, 6.0, 0,
                            [shell, ok1, ok2, m_ferro, neel, m_res,
                             arrow0](bool ok3) {
                                const double arrow1 =
                                    shell->sn()->arrow_mean();
                                const bool pass =
                                    ok1 && ok2 && ok3 && m_ferro > 0.85 &&
                                    neel > 0.85 && m_res < 0.35 &&
                                    arrow0 > 0.95 && arrow1 < 0.9;
                                std::fprintf(
                                    stderr,
                                    "selftest-spins: ferro |M| %.2f, "
                                    "Neel %.2f (|M| %.2f), exact arrows "
                                    "%.2f -> %.2f  [%s]\n",
                                    m_ferro, neel, m_res, arrow0, arrow1,
                                    pass ? "PASS" : "FAIL");
                                shell->request_exit(pass ? 0 : 1);
                            });
                    });
            });
        });
    }

    // Bouncer arc (main forces --scene=bouncer1d): the boot relax lands
    // in the soft-floor Airy window; a drop from height carries E ~ g h.
    // The exact Airy SPACING is pinned by tests/bouncer1d_test.cpp.
    if (shell->has_arg("--selftest-bouncer")) {
        shell->sched().after(1000, [shell] {
            auto* bo = shell->bo();
            if (bo == nullptr) {
                std::fprintf(stderr, "selftest-bouncer: no api  [FAIL]\n");
                shell->request_exit(1);
                return;
            }
            const double e_g = bo->energy();  // boot = relaxed ground
            const double e1 = bo->airy_e1();
            bo->drop();
            const double e_d = bo->energy();
            const bool ground_ok = e_g < e1 && e_g > e1 - 0.35;
            const bool drop_ok = e_d > 0.85 * 80.0 && e_d < 1.15 * 80.0;
            const bool pass = ground_ok && drop_ok;
            std::fprintf(stderr,
                         "selftest-bouncer: ground %.3f (Airy %.3f, soft "
                         "floor), drop E %.1f (g h = 80)  [%s]\n",
                         e_g, e1, e_d, pass ? "PASS" : "FAIL");
            shell->request_exit(pass ? 0 : 1);
        });
    }

    // QPC arc (main forces --scene=qpc2d): the staircase foot vs the
    // first channel at scene scale. The full 4-point staircase is pinned
    // by tests/qpc2d_test.cpp.
    if (shell->has_arg("--selftest-qpc2d")) {
        shell->sched().after(1000, [shell] {
            selftest_scene_wait_running(shell, "qpc2d", 0, [shell](
                                                               bool runs) {
                auto* qp = shell->qp();
                if (!runs || qp == nullptr) {
                    std::fprintf(stderr, "selftest-qpc2d: scene not "
                                         "running or no api  [FAIL]\n");
                    shell->request_exit(1);
                    return;
                }
                const double t_run = 60.0;
                shell->set_time_scale(16);
                qp->set_gap(2.0);  // below the first mode
                selftest_wait_sim_time(shell, t_run, 0, [shell,
                                                         t_run](bool ok1) {
                    const double t_closed = shell->qp()->transmitted();
                    shell->qp()->set_gap(4.5);  // one channel open
                    selftest_wait_sim_time(
                        shell, t_run, 0,
                        [shell, ok1, t_closed](bool ok2) {
                            const double t_open =
                                shell->qp()->transmitted();
                            const bool pass = ok1 && ok2 &&
                                              t_open > 0.02 &&
                                              t_closed < 0.3 * t_open;
                            std::fprintf(
                                stderr,
                                "selftest-qpc2d: transmitted gap 2.0 = "
                                "%.3f, gap 4.5 = %.3f  [%s]\n",
                                t_closed, t_open, pass ? "PASS" : "FAIL");
                            shell->request_exit(pass ? 0 : 1);
                        });
                });
            });
        });
    }

    // Carpet arc (main forces --scene=carpet1d): scrambled mid-carpet,
    // full revival at T_rev = L^2/pi -- maxima tracked at row cadence by
    // the director (frame polling would miss the ~2 au peak).
    if (shell->has_arg("--selftest-carpet")) {
        shell->sched().after(1000, [shell] {
            selftest_scene_wait_running(shell, "carpet1d", 0, [shell](
                                                                  bool runs) {
                auto* cp = shell->cp();
                if (!runs || cp == nullptr) {
                    std::fprintf(stderr, "selftest-carpet: scene not "
                                         "running or no api  [FAIL]\n");
                    shell->request_exit(1);
                    return;
                }
                const double t_rev = cp->revival_time();
                shell->set_time_scale(16);
                selftest_wait_sim_time(
                    shell, 1.05 * t_rev, 0, [shell](bool ok1) {
                        auto* c2 = shell->cp();
                        const double mid = c2->mid_scramble_max();
                        const double best = c2->best_revival();
                        // A few-mode ring packet has LARGE fractional
                        // revivals (measured 0.63 mid-carpet): the honest
                        // gate is that the full revival stands clear
                        // above every mid recurrence, near unity.
                        const bool pass = ok1 && best > 0.95 &&
                                          mid < 0.85 && best > mid + 0.2;
                        std::fprintf(stderr,
                                     "selftest-carpet: mid-carpet max "
                                     "%.2f, revival %.2f (T_rev %.0f au)  "
                                     "[%s]\n",
                                     mid, best, c2->revival_time(),
                                     pass ? "PASS" : "FAIL");
                        shell->request_exit(pass ? 0 : 1);
                    });
            });
        });
    }

    // Cat + photon-loss arc (harmonic1d): the cat's <n> bleeds at kappa
    // and photons actually click (jumps fire); the parity-flip contract
    // itself is pinned grid-exactly in tests/mcwf1d_test.cpp.
    if (shell->has_arg("--selftest-cat")) {
        shell->sched().after(1000, [shell] {
            selftest_scene_wait_running(shell, "harmonic1d", 0, [shell](
                                                                    bool
                                                                        runs) {
                auto* ln = shell->ln();
                if (!runs || ln == nullptr) {
                    std::fprintf(stderr, "selftest-cat: scene not running "
                                         "or no api  [FAIL]\n");
                    shell->request_exit(1);
                    return;
                }
                ln->cat();
                const double w = ln->omega();
                const double n0 = ln->level_energy() / w - 0.5;
                ln->toggle_loss();
                const double t_mark = shell->director().sim_time();
                shell->set_time_scale(16);
                selftest_wait_sim_time(
                    shell, t_mark + 80.0, 0, [shell, w, n0](bool ok1) {
                        auto* l2 = shell->ln();
                        const double n1 = l2->level_energy() / w - 0.5;
                        const long long jumps = l2->jump_count();
                        const bool pass = ok1 && n0 > 4.0 &&
                                          n1 < 0.35 * n0 && jumps >= 3;
                        std::fprintf(stderr,
                                     "selftest-cat: <n> %.2f -> %.2f over "
                                     "80 au (kappa 0.05), photons %lld  "
                                     "[%s]\n",
                                     n0, n1, jumps, pass ? "PASS" : "FAIL");
                        shell->request_exit(pass ? 0 : 1);
                    });
            });
        });
    }

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
                // The quenched state is a superposition: it must ladder
                // through the truncated Fock basis (exact coefficient
                // action -- the raw spectral chain is useless at this
                // grid's k_max) with the band as its cap.
                ln->set_omega(1.0);
                const int cap_mix = ln->max_level();
                const bool fock_ok =
                    ln->level() == -1 && cap_mix >= 50 && ln->ladder(true);
                // Back on an eigenstate the cap is the representability
                // ceiling of the widened box -- far above the old raw
                // chain's teens.
                shell->press('2');
                const int cap_eigen = ln->max_level();
                const bool ceiling_ok = ln->level() == 0 && cap_eigen >= 100;
                const bool pass = up_ok && e_ok && down_ok && refuse_ok &&
                                  mix_ok && quench_ok && trip_ok && fock_ok &&
                                  ceiling_ok;
                std::fprintf(stderr,
                             "selftest-ladder1d: up %d (E2 = %.4f Ha, ok %d), "
                             "down %d, ground-refuse %d, superposition %d, "
                             "quench %d, stable-trip20 %d, fock-mix %d "
                             "(band %d), eigen-ceiling %d (cap %d)  [%s]\n",
                             up_ok, e2, e_ok, down_ok, refuse_ok, mix_ok,
                             quench_ok, trip_ok, fock_ok, cap_mix, ceiling_ok,
                             cap_eigen, pass ? "PASS" : "FAIL");
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

    // Double-well arc (main forces --scene=doublewell1d): psi_L must start
    // left and, within ~1.5 transfer periods at time_scale 16, appear on
    // the right with P_R > 0.8 -- the ammonia-inversion oscillation, live.
    if (shell->has_arg("--selftest-dw1d")) {
        shell->sched().after(1000, [shell] {
            selftest_scene_wait_running(shell, "doublewell1d", 0, [shell](
                                                                     bool runs) {
                auto* dw = shell->dw();
                if (!runs || dw == nullptr) {
                    std::fprintf(stderr, "selftest-dw1d: scene not running "
                                         "or no api  [FAIL]\n");
                    shell->request_exit(1);
                    return;
                }
                const double de = dw->splitting();
                const bool band_ok = de > 3e-3 && de < 5e-2;
                const bool left_ok = dw->p_left() > 0.9;
                shell->set_time_scale(16);
                auto best = std::make_shared<double>(0.0);
                auto probe = std::make_shared<int>(-1);
                *probe = shell->sched().every(500, [shell, best, probe] {
                    *best = std::max(*best, shell->dw()->p_right());
                    if (*best > 0.8) {
                        shell->sched().cancel(*probe);
                        std::fprintf(stderr, "selftest-dw1d: transferred, "
                                             "max P_R = %.3f  [PASS]\n",
                                     *best);
                        shell->request_exit(0);
                    }
                });
                shell->sched().after(30000, [shell, best, probe, de, band_ok,
                                             left_ok] {
                    shell->sched().cancel(*probe);
                    const bool pass = band_ok && left_ok && *best > 0.8;
                    std::fprintf(stderr,
                                 "selftest-dw1d: dE = %.2e (band %d), start "
                                 "left %d, max P_R = %.3f  [%s]\n",
                                 de, band_ok, left_ok, *best,
                                 pass ? "PASS" : "FAIL");
                    shell->request_exit(pass ? 0 : 1);
                });
            });
        });
    }

    // Reflectionless arc (main forces --scene=ptwell1d): the sech^2 well
    // must show max R below 5e-3 after the transit; the equal square well
    // (Key W) must reflect visibly. SIM-TIME polled, not wall-clocked: a
    // 64k-point step costs real milliseconds, so the transit (~110 au)
    // takes however long it takes (repo lesson: fixed deadlines lie).
    if (shell->has_arg("--selftest-pt1d")) {
        shell->sched().after(1000, [shell] {
            selftest_scene_wait_running(shell, "ptwell1d", 0, [shell](
                                                                  bool runs) {
                if (!runs || shell->rf() == nullptr) {
                    std::fprintf(stderr, "selftest-pt1d: scene not running "
                                         "or no api  [FAIL]\n");
                    shell->request_exit(1);
                    return;
                }
                shell->set_time_scale(16);
                selftest_wait_sim_time(shell, 160.0, 0, [shell](bool ok1) {
                    const double r_pt = shell->rf()->reflected_max();
                    const double t_mark = shell->director().sim_time();
                    shell->rf()->toggle_well();  // -> square, relaunch
                    selftest_wait_sim_time(shell, t_mark + 160.0, 0,
                                           [shell, ok1, r_pt](bool ok2) {
                        const double r_sq = shell->rf()->reflected_max();
                        const bool pass =
                            ok1 && ok2 && r_pt < 5e-3 && r_sq > 3e-2;
                        std::fprintf(stderr,
                                     "selftest-pt1d: R(sech^2) = %.4f, "
                                     "R(square) = %.4f (timely %d %d)  [%s]\n",
                                     r_pt, r_sq, ok1, ok2,
                                     pass ? "PASS" : "FAIL");
                        shell->request_exit(pass ? 0 : 1);
                    });
                });
            });
        });
    }

    // Morse arc (main forces --scene=morse1d): exactly 6 bound levels, the
    // ladder climbs to the top and refuses past it, and the top gap is
    // visibly smaller than the bottom gap (the anharmonic signature).
    if (shell->has_arg("--selftest-morse1d")) {
        shell->sched().after(1000, [shell] {
            selftest_scene_wait_running(shell, "morse1d", 0, [shell](
                                                                 bool runs) {
                auto* mo = shell->mo();
                if (!runs || mo == nullptr) {
                    std::fprintf(stderr, "selftest-morse1d: scene not "
                                         "running or no api  [FAIL]\n");
                    shell->request_exit(1);
                    return;
                }
                const bool count_ok = mo->bound_count() == 6;
                const double e0 = mo->level_energy();
                mo->jump(true);
                const double e1 = mo->level_energy();
                bool climb_ok = true;
                while (mo->level() + 1 < mo->bound_count()) {
                    climb_ok = climb_ok && mo->jump(true);
                }
                const double e_top = mo->level_energy();
                mo->jump(false);
                const double e_below = mo->level_energy();
                const bool refuse_ok = [&] {
                    while (mo->level() + 1 < mo->bound_count()) {
                        mo->jump(true);
                    }
                    return !mo->jump(true);  // past the top: dissociation
                }();
                const double gap_bot = e1 - e0;
                const double gap_top = e_top - e_below;
                const bool anharm_ok = gap_top < 0.6 * gap_bot;
                const bool pass =
                    count_ok && climb_ok && refuse_ok && anharm_ok;
                std::fprintf(stderr,
                             "selftest-morse1d: bound %d, climb %d, "
                             "top-refuse %d, gaps %.4f -> %.4f (anharm %d)  "
                             "[%s]\n",
                             mo->bound_count(), climb_ok, refuse_ok, gap_bot,
                             gap_top, anharm_ok, pass ? "PASS" : "FAIL");
                shell->request_exit(pass ? 0 : 1);
            });
        });
    }

    // 2D double-slit + AB arc (main forces --scene=doubleslit2d): fly the
    // packet through the pierced wall and let the screen integrate the
    // arrivals, at Phi = 0 (bright axis) and Phi = pi (Chambers shift:
    // dark axis, first-dark position turns bright). SIM-TIME polled.
    if (shell->has_arg("--selftest-doubleslit2d")) {
        shell->sched().after(1000, [shell] {
            selftest_scene_wait_running(shell, "doubleslit2d", 0, [shell](
                                                                      bool
                                                                          runs) {
                auto* sl = shell->sl();
                if (!runs || sl == nullptr) {
                    std::fprintf(stderr, "selftest-doubleslit2d: scene not "
                                         "running or no api  [FAIL]\n");
                    shell->request_exit(1);
                    return;
                }
                const double pi = 3.14159265358979323846;
                // First dark fringe: path difference lambda/2 at the
                // screen, y = lambda L / (2 d). ONE electron per shot
                // (user order): k0 = 1 long-wavelength packet, transit
                // ~80 au + tail -- t_run covers a full shot.
                const double lam = 2.0 * pi / 1.0;   // k0 = 1
                const double ell = 45.0 - 0.75;      // wall mid -> screen
                const double yd = lam * ell / (2.0 * sl->separation());
                const double t_run = 120.0;
                shell->set_time_scale(16);
                selftest_wait_sim_time(shell, t_run, 0, [shell, yd, pi,
                                                         t_run](bool ok1) {
                    const double b0 = shell->sl()->screen_at(0.0);
                    const double d0 = shell->sl()->screen_at(yd);
                    // Fire a SECOND electron: the screen must keep the
                    // first shot's arrivals and roughly double on axis.
                    shell->sl()->refire();
                    selftest_wait_sim_time(shell, 2.0 * t_run, 0, [shell,
                                                                   yd, pi,
                                                                   t_run, ok1,
                                                                   b0, d0](
                                                                      bool
                                                                          okr) {
                        const double b2 = shell->sl()->screen_at(0.0);
                        shell->sl()->set_flux(pi);  // full reset + refire
                        selftest_wait_sim_time(
                            shell, t_run, 0,
                            [shell, yd, ok1, okr, b0, d0, b2](bool ok2) {
                                const double bpi = shell->sl()->screen_at(0.0);
                                const double dpi = shell->sl()->screen_at(yd);
                                const bool young_ok = d0 < 0.35 * b0;
                                const bool accum_ok =
                                    b2 > 1.6 * b0 && b2 < 2.4 * b0;
                                const bool ab_ok =
                                    bpi < 0.35 * b0 && dpi > 0.5 * d0 &&
                                    dpi > 0.35 * b0;
                                const bool pass = ok1 && okr && ok2 &&
                                                  b0 > 0.0 && young_ok &&
                                                  accum_ok && ab_ok;
                                std::fprintf(
                                    stderr,
                                    "selftest-doubleslit2d: screen(axis/dark) "
                                    "Phi=0: %.3e/%.3e, shot2 x%.2f, "
                                    "Phi=pi: %.3e/%.3e  [%s]\n",
                                    b0, d0, b0 > 0.0 ? b2 / b0 : 0.0, bpi,
                                    dpi, pass ? "PASS" : "FAIL");
                                shell->request_exit(pass ? 0 : 1);
                            });
                    });
                });
            });
        });
    }

    // Landau arc (main forces --scene=landau2d): ride one cyclotron
    // period -- antipode at T/2, home at T -- SIM-TIME polled.
    if (shell->has_arg("--selftest-landau")) {
        shell->sched().after(1000, [shell] {
            selftest_scene_wait_running(shell, "landau2d", 0, [shell](
                                                                  bool runs) {
                auto* la = shell->la();
                if (!runs || la == nullptr) {
                    std::fprintf(stderr, "selftest-landau: scene not "
                                         "running or no api  [FAIL]\n");
                    shell->request_exit(1);
                    return;
                }
                const double pi = 3.14159265358979323846;
                const double r = la->radius_pred();
                const double period = 2.0 * pi / la->omega_c();
                shell->set_time_scale(16);
                // The DIRECTOR records the antipode/closure distances at
                // the actual crossings (step-chunk granularity); the arc
                // only has to wait PAST the period and read them --
                // sim-time polls are far too coarse for orbital phase.
                selftest_wait_sim_time(shell, 1.1 * period, 0, [shell,
                                                                r](bool ok1) {
                    const double anti = shell->la()->antipode_dist();
                    const double home = shell->la()->closure_dist();
                    const bool pass = ok1 && anti >= 0.0 && home >= 0.0 &&
                                      anti < 0.25 * r && home < 0.25 * r;
                    std::fprintf(stderr,
                                 "selftest-landau: r = %.2f, antipode "
                                 "miss %.2f, closure miss %.2f  [%s]\n",
                                 r, anti, home, pass ? "PASS" : "FAIL");
                    shell->request_exit(pass ? 0 : 1);
                });
            });
        });
    }

    // Bloch arc (main forces --scene=bloch1d): one Bloch period under the
    // tilt -- the packet must stay BOUNDED (no runaway) and come home,
    // while a free particle would have fallen F T_B^2 / 2 away.
    if (shell->has_arg("--selftest-bloch")) {
        shell->sched().after(1000, [shell] {
            selftest_scene_wait_running(shell, "bloch1d", 0, [shell](
                                                                 bool runs) {
                auto* bl = shell->bl();
                if (!runs || bl == nullptr) {
                    std::fprintf(stderr, "selftest-bloch: scene not "
                                         "running or no api  [FAIL]\n");
                    shell->request_exit(1);
                    return;
                }
                const double x0 = bl->mean_x();
                const double t_b = bl->bloch_period();
                const double free_fall =
                    0.5 * bl->force() * t_b * t_b;
                shell->set_time_scale(16);
                selftest_wait_sim_time(shell, t_b, 0, [shell, x0, t_b,
                                                       free_fall](bool ok1) {
                    const double dx =
                        std::abs(shell->bl()->mean_x() - x0);
                    const double exc = shell->bl()->excursion();
                    const bool moved_ok = exc > 0.3;      // it DID slosh
                    const bool bound_ok =
                        exc < 0.15 * free_fall;           // never ran away
                    const bool home_ok = dx < 2.0;        // and came home
                    const bool pass =
                        ok1 && moved_ok && bound_ok && home_ok;
                    std::fprintf(stderr,
                                 "selftest-bloch: T_B = %.0f, excursion "
                                 "%.2f (free fall %.0f), |dx(T_B)| = %.2f  "
                                 "[%s]\n",
                                 t_b, exc, free_fall, dx,
                                 pass ? "PASS" : "FAIL");
                    shell->request_exit(pass ? 0 : 1);
                });
            });
        });
    }

    // Corral arc (main forces --scene=corral2d): the boot relax must
    // capture the ground INSIDE the fence with the J0-mode energy scale.
    // Anderson arc (main forces --scene=anderson1d): the conductance
    // contract at scene scale -- the disordered wire insulates, the clean
    // wire conducts. Mirrors anderson1d_test.
    if (shell->has_arg("--selftest-anderson")) {
        shell->sched().after(1000, [shell] {
            selftest_scene_wait_running(shell, "anderson1d", 0, [shell](
                                                                    bool
                                                                        runs) {
                auto* an = shell->an();
                if (!runs || an == nullptr) {
                    std::fprintf(stderr, "selftest-anderson: scene not "
                                         "running or no api  [FAIL]\n");
                    shell->request_exit(1);
                    return;
                }
                const double t_run = 110.0;
                shell->set_time_scale(16);
                selftest_wait_sim_time(shell, t_run, 0, [shell,
                                                         t_run](bool ok1) {
                    const double t_dis = shell->an()->transmitted();
                    shell->an()->set_disorder(0.0);  // clean wire, refires
                    selftest_wait_sim_time(
                        shell, t_run, 0, [shell, ok1, t_dis](bool ok2) {
                            const double t_clean =
                                shell->an()->transmitted();
                            const bool pass = ok1 && ok2 && t_clean > 0.7 &&
                                              t_dis < 0.3 * t_clean;
                            std::fprintf(
                                stderr,
                                "selftest-anderson: transmitted disordered "
                                "%.3f, clean %.3f  [%s]\n",
                                t_dis, t_clean, pass ? "PASS" : "FAIL");
                            shell->request_exit(pass ? 0 : 1);
                        });
                });
            });
        });
    }

    // Billiard arc (main forces --scene=billiard2d): the caustic contract
    // at scene scale -- circle keeps the center dark (|L| conserved),
    // the stadium fills it (chaos). Mirrors billiard2d_test.
    if (shell->has_arg("--selftest-billiard")) {
        shell->sched().after(1000, [shell] {
            selftest_scene_wait_running(shell, "billiard2d", 0, [shell](
                                                                    bool
                                                                        runs) {
                auto* bd = shell->bd();
                if (!runs || bd == nullptr) {
                    std::fprintf(stderr, "selftest-billiard: scene not "
                                         "running or no api  [FAIL]\n");
                    shell->request_exit(1);
                    return;
                }
                const double t_run = 90.0;
                shell->set_time_scale(16);
                selftest_wait_sim_time(shell, t_run, 0, [shell,
                                                         t_run](bool ok1) {
                    const double f_circle = shell->bd()->avg_center_fraction();
                    shell->bd()->toggle_shape();  // stadium, avg + clock reset
                    selftest_wait_sim_time(
                        shell, t_run, 0, [shell, ok1, f_circle](bool ok2) {
                            const double f_stad =
                                shell->bd()->avg_center_fraction();
                            const bool pass = ok1 && ok2 && f_circle >= 0.0 &&
                                              f_circle < 0.30 &&
                                              f_stad > 2.0 * f_circle;
                            std::fprintf(
                                stderr,
                                "selftest-billiard: center/interior circle "
                                "%.3f, stadium %.3f  [%s]\n",
                                f_circle, f_stad, pass ? "PASS" : "FAIL");
                            shell->request_exit(pass ? 0 : 1);
                        });
                });
            });
        });
    }

    if (shell->has_arg("--selftest-corral")) {
        shell->sched().after(1000, [shell] {
            auto* cr = shell->cr();
            if (cr == nullptr) {
                std::fprintf(stderr, "selftest-corral: scene not "
                                     "running or no api  [FAIL]\n");
                shell->request_exit(1);
                return;
            }
            // The corral BOOTS inside its relax (sim time frozen), so the
            // generic sim-time running gate cannot apply: poll the capture
            // directly, bounded (annealed 512^2 relax takes a while).
            shell->set_time_scale(16);
            auto poll = std::make_shared<int>(-1);
            auto tries = std::make_shared<int>(0);
            *poll = shell->sched().every(1000, [shell, poll, tries] {
                auto* c = shell->cr();
                if (c->relaxing() || c->captured() < 1) {
                    if (++*tries >= 240) {
                        shell->sched().cancel(*poll);
                        std::fprintf(stderr,
                                     "selftest-corral: relax never "
                                     "converged  [FAIL]\n");
                        shell->request_exit(1);
                    }
                    return;
                }
                shell->sched().cancel(*poll);
                const double conf = c->confinement();
                const double e = c->energy(0);
                const double r = c->radius();
                const double e_hard =
                    2.405 * 2.405 / (2.0 * c->mass() * r * r);
                const bool pass = conf > 0.85 && e > 0.6 * e_hard &&
                                  e < 2.0 * e_hard;
                std::fprintf(stderr,
                             "selftest-corral: E0 = %.4f (J0 scale "
                             "%.4f), P(inside) = %.2f  [%s]\n",
                             e, e_hard, conf, pass ? "PASS" : "FAIL");
                shell->request_exit(pass ? 0 : 1);
            });
        });
    }

    // Quantum-dot arc (main forces --scene=qdot2d): the relaxed ground at
    // the boot (w0, B) must land on the Fock-Darwin Omega; then B = 0
    // must land on w0 -- the field dependence verified end to end.
    if (shell->has_arg("--selftest-qdot")) {
        shell->sched().after(1000, [shell] {
            selftest_scene_wait_running(shell, "qdot2d", 0, [shell](
                                                                bool runs) {
                auto* qd = shell->qd();
                if (!runs || qd == nullptr) {
                    std::fprintf(stderr, "selftest-qdot: scene not "
                                         "running or no api  [FAIL]\n");
                    shell->request_exit(1);
                    return;
                }
                shell->set_time_scale(16);
                auto poll = std::make_shared<int>(-1);
                *poll = shell->sched().every(1000, [shell, poll] {
                    auto* q = shell->qd();
                    if (q->relaxing()) {
                        return;
                    }
                    shell->sched().cancel(*poll);
                    const double e1 = q->energy_meas();
                    const double o1 = q->energy_pred();
                    q->set_field(0.0);  // re-relax at B = 0
                    auto poll2 = std::make_shared<int>(-1);
                    *poll2 = shell->sched().every(1000, [shell, poll2, e1,
                                                        o1] {
                        auto* q2 = shell->qd();
                        if (q2->relaxing()) {
                            return;
                        }
                        shell->sched().cancel(*poll2);
                        const double e2 = q2->energy_meas();
                        const double o2 = q2->energy_pred();
                        const bool pass =
                            std::abs(e1 - o1) < 0.03 * o1 &&
                            std::abs(e2 - o2) < 0.03 * o2 && o1 > o2;
                        std::fprintf(
                            stderr,
                            "selftest-qdot: E(B) = %.4f vs Omega %.4f, "
                            "E(0) = %.4f vs w0 %.4f  [%s]\n",
                            e1, o1, e2, o2, pass ? "PASS" : "FAIL");
                        shell->request_exit(pass ? 0 : 1);
                    });
                });
            });
        });
    }

    // H2+ arc (main forces --scene=h2plus): sigma_g then sigma_u at the
    // equilibrium R, then the stretched geometry -- asserting the bond:
    // E_total(R_eq) < E_total(R_far), and sigma_u above sigma_g. State
    // preparation is ITP over frames: poll prepared(k), never wall-clock.
    if (shell->has_arg("--selftest-h2p")) {
        shell->sched().after(1000, [shell] {
            selftest_scene_wait_running(shell, "h2plus", 0, [shell](bool runs) {
                auto* ml = shell->ml();
                if (!runs || ml == nullptr) {
                    std::fprintf(stderr, "selftest-h2p: scene not running or "
                                         "no api  [FAIL]\n");
                    shell->request_exit(1);
                    return;
                }
                shell->set_time_scale(16);
                ml->prepare(1);  // atlas: 1sigma_g, then 1sigma_u*
                auto poll = std::make_shared<int>(-1);
                *poll = shell->sched().every(2000, [shell, poll] {
                    auto* m = shell->ml();
                    if (!m->prepared(1)) {
                        return;
                    }
                    shell->sched().cancel(*poll);
                    // R is fixed at equilibrium (no bond scan): check the
                    // bonding/antibonding ordering AND that the total energy
                    // binds vs the dissociation limit H(1s) + p = -0.5 Ha.
                    const double e_g = m->energy(0);
                    const double e_u = m->energy(1);
                    const double et = e_g + m->nuclear_repulsion();
                    const bool order_ok = e_u > e_g;
                    const bool bond_ok = et < -0.5;
                    const bool pass = order_ok && bond_ok;
                    std::fprintf(stderr,
                                 "selftest-h2p: E_g = %.4f, E_u = %.4f, "
                                 "E_tot = %.4f < -0.5 (bound)?  [%s]\n",
                                 e_g, e_u, et, pass ? "PASS" : "FAIL");
                    shell->request_exit(pass ? 0 : 1);
                });
                shell->sched().after(60000, [shell, poll] {
                    shell->sched().cancel(*poll);
                    if (!shell->ml()->prepared(1)) {
                        std::fprintf(stderr, "selftest-h2p: chain TIMEOUT  "
                                             "[FAIL]\n");
                        shell->request_exit(1);
                    }
                });
            });
        });
    }

    // H2+ known-orbitals arc (main forces --scene=h2plus): the deflated
    // chain climbs to 1pi_u (state 2), and the three MO energies must order
    // 1sigma_g < 1sigma_u* < 1pi_u. Then a random seed must STAY bound
    // (P(r<8) high) -- an arbitrary normalized state that evolves, not junk.
    if (shell->has_arg("--selftest-h2p-orbitals")) {
        shell->sched().after(1000, [shell] {
            selftest_scene_wait_running(shell, "h2plus", 0, [shell](bool runs) {
                auto* ml = shell->ml();
                if (!runs || ml == nullptr) {
                    std::fprintf(stderr, "selftest-h2p-orbitals: scene not "
                                         "running or no api  [FAIL]\n");
                    shell->request_exit(1);
                    return;
                }
                shell->set_time_scale(16);
                ml->prepare(2);  // chain: sigma_g, sigma_u*, pi_u
                auto poll = std::make_shared<int>(-1);
                *poll = shell->sched().every(2000, [shell, poll] {
                    auto* m = shell->ml();
                    if (!m->prepared(2)) {
                        return;
                    }
                    shell->sched().cancel(*poll);
                    const double e0 = m->energy(0);
                    const double e1 = m->energy(1);
                    const double e2 = m->energy(2);
                    const bool order_ok = e0 < e1 && e1 < e2;
                    std::fprintf(stderr,
                                 "selftest-h2p-orbitals: E(1sg) = %.4f < "
                                 "E(1su*) = %.4f < E(1pu) = %.4f?  [%s]\n",
                                 e0, e1, e2, order_ok ? "PASS" : "FAIL");
                    shell->request_exit(order_ok ? 0 : 1);
                });
                shell->sched().after(300000, [shell, poll] {
                    shell->sched().cancel(*poll);
                    if (!shell->ml()->prepared(2)) {
                        std::fprintf(stderr, "selftest-h2p-orbitals: chain "
                                             "TIMEOUT  [FAIL]\n");
                        shell->request_exit(1);
                    }
                });
            });
        });
    }

    // Stripped-benzene arc (main forces --scene=benzene): the first
    // electron of C6H6^41+ over bare nuclei (the REAL uniform geometry;
    // no counterfactual knobs). The three prepared states must form a
    // DEEP quasi-degenerate carbon-core band (no ordering or gap claims:
    // deflated ITP finds band members in arbitrary order).
    if (shell->has_arg("--selftest-benzene")) {
        shell->sched().after(1000, [shell] {
            selftest_scene_wait_running(shell, "benzene", 0, [shell](bool runs) {
                auto* ml = shell->ml();
                if (!runs || ml == nullptr) {
                    std::fprintf(stderr, "selftest-benzene: scene not running "
                                         "or no api  [FAIL]\n");
                    shell->request_exit(1);
                    return;
                }
                shell->set_time_scale(16);
                ml->prepare(2);  // chain: ground, pair member 1, pair member 2
                auto poll = std::make_shared<int>(-1);
                *poll = shell->sched().every(2000, [shell, poll] {
                    auto* m = shell->ml();
                    if (!m->prepared(2)) {
                        return;
                    }
                    shell->sched().cancel(*poll);
                    const double e0 = m->energy(0);
                    const double e1 = m->energy(1);
                    const double e2 = m->energy(2);
                    const double lo = std::min(e0, std::min(e1, e2));
                    const double hi = std::max(e0, std::max(e1, e2));
                    const bool band_ok = hi < -8.0 && (hi - lo) < 1.0;
                    std::fprintf(stderr,
                                 "selftest-benzene: core band [%.3f, %.3f] "
                                 "Ha (width %.3f)  [%s]\n",
                                 lo, hi, hi - lo, band_ok ? "PASS" : "FAIL");
                    if (!band_ok) {
                        shell->request_exit(1);
                        return;
                    }
                    // Real-time containment: a prepared CORE state must
                    // STAY on the ring under real-time stepping (the step
                    // accuracy contract -- a dt too coarse for the Z=6
                    // regularized well heats the state over the whole box).
                    shell->set_real_time();
                    shell->set_time_scale(16);
                    const double t0 = shell->director().sim_time();
                    selftest_wait_sim_time(shell, t0 + 5.0, 0, [shell](
                                                                   bool ok_t) {
                        auto* m2 = shell->ml();
                        const double c = m2->containment(6.0);
                        const bool pass = ok_t && c > 0.9;
                        std::fprintf(stderr,
                                     "selftest-benzene: P(r<6) = %.3f after "
                                     "5 au real time  [%s]\n",
                                     c, pass ? "PASS" : "FAIL");
                        shell->request_exit(pass ? 0 : 1);
                    });
                });
                shell->sched().after(300000, [shell, poll] {
                    shell->sched().cancel(*poll);
                    if (!shell->ml()->prepared(2)) {
                        std::fprintf(stderr, "selftest-benzene: chain TIMEOUT "
                                             " [FAIL]\n");
                        shell->request_exit(1);
                    }
                });
            });
        });
    }

    // Rutherford arc (main forces --scene=rutherford3d): a packet fired
    // head-on at the repulsive Coulomb center turns around near the classical
    // closest approach r_min = 2Z/E (never reaching the core) and backscatters
    // -- the barrier that revealed the nucleus. Poll the director's probes.
    if (shell->has_arg("--selftest-rutherford")) {
        run_when_manifold_ready(shell, [shell] {
            auto* rf = shell->director().rutherford();
            if (rf == nullptr) {
                std::fprintf(stderr, "selftest-rutherford: no api  [FAIL]\n");
                shell->request_exit(1);
                return;
            }
            shell->set_time_scale(16);
            shell->sched().after(120000, [shell] {
                auto* r = shell->director().rutherford();
                const double rmin = r->turning_point();      // classical 2Z/E
                const double closest = r->closest_approach();  // min <r> seen
                const double back = r->backscattered_fraction();
                // The packet turned around near r_min: it did NOT reach the
                // core (closest > 0.5 r_min) and did approach (closest <
                // launch distance), and a real fraction came back upstream.
                const bool approached = closest > 0.4 * rmin && closest < 28.0;
                const bool reflected = back > 0.1;
                const bool pass = approached && reflected;
                std::fprintf(stderr,
                             "selftest-rutherford: r_min = %.1f, closest <r> "
                             "= %.1f, backscatter = %.2f  [%s]\n",
                             rmin, closest, back, pass ? "PASS" : "FAIL");
                shell->request_exit(pass ? 0 : 1);
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
