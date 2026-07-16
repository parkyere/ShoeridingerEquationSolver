module;
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
export module ses.app.selftest_arcs;
export import ses.app.manifold_spec;


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
