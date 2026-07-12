#pragma once

// Verification + selftest arcs: every --dump-frame* and --selftest-* arc,
// registered against the live viewport. Templated on the viewport type so
// the shell class stays private to main.cpp. Each arc waits for the startup
// atlas build (run_when_manifold_ready) and then CHAINS timers, so a slower
// GPU stretches the run instead of false-failing a wall-clock verdict.

#include "manifold_spec.hpp"

#include <QApplication>
#include <QImage>
#include <QString>
#include <QTimer>

#include <cstdio>
#include <memory>

namespace ses_shell {

// Selftest helper: poll until the startup atlas build has produced the
// channel table, then run the arc (slower GPUs just stretch the wait).
template <typename ViewportT, typename F>
void run_when_manifold_ready(ViewportT* viewport, F fn) {
    if (viewport->manifold_ready()) {
        fn();
        return;
    }
    QTimer::singleShot(500, viewport,
                       [viewport, fn] { run_when_manifold_ready(viewport, fn); });
}

template <typename ViewportT>
void register_verification_arcs(QApplication& app, ViewportT* viewport) {
    // Render verification: grab the composited frame to frame_dump.bmp and
    // exit; grabFramebuffer verifies the whole path (ses_vk scene + Qt blit)
    // end to end. BMP because the lean static Qt has no png feature.
    if (app.arguments().contains(QStringLiteral("--dump-frame"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            QTimer::singleShot(2000, viewport, [viewport, &app] {
                const QImage frame = viewport->grabFramebuffer();
                const bool ok = !frame.isNull() &&
                                frame.save(QStringLiteral("frame_dump.bmp"), "BMP");
                std::fprintf(stderr, "dump-frame: %dx%d  [%s]\n", frame.width(),
                             frame.height(), ok ? "PASS" : "FAIL");
                app.exit(ok ? 0 : 1);
            });
        });
    }
    // Same, from INSIDE the box: the volume pass rasterizes the proxy's back
    // faces (front-face culled), which only an interior eye exercises.
    if (app.arguments().contains(QStringLiteral("--dump-frame-near"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            std::fprintf(stderr, "dump-frame-near: manifold ready, zooming in\n");
            viewport->debug_set_camera_distance(4.0);
            QTimer::singleShot(2000, viewport, [viewport, &app] {
                std::fprintf(stderr, "dump-frame-near: grabbing\n");
                const QImage frame = viewport->grabFramebuffer();
                const bool ok = !frame.isNull() &&
                                frame.save(QStringLiteral("frame_dump_near.bmp"), "BMP");
                std::fprintf(stderr, "dump-frame-near: %dx%d  [%s]\n", frame.width(),
                             frame.height(), ok ? "PASS" : "FAIL");
                app.exit(ok ? 0 : 1);
            });
        });
    }

    // Decay arc: prepare 2p, return to real time, require >= 1 quantum jump
    // (after the jump the atom sits in 1s, so one photon is the expected
    // outcome). Photon verdicts count from a baseline at the arc's start.
    if (app.arguments().contains(QStringLiteral("--selftest-decay"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            viewport->relax_to_excited();  // caches ready: no block
            QTimer::singleShot(13500, viewport, [viewport, &app] {
                const long long baseline = viewport->photon_count();
                viewport->set_real_time();  // decay is already armed
                QTimer::singleShot(30000, viewport, [viewport, &app, baseline] {
                    const long long fresh = viewport->photon_count() - baseline;
                    std::fprintf(stderr, "selftest-decay: photons = %lld  [%s]\n",
                                 fresh, fresh >= 1 ? "PASS" : "FAIL");
                    app.exit(fresh >= 1 ? 0 : 1);
                });
            });
        });
    }

    // Energy-measurement arc: relax to 1s (decay OFF so the state stays
    // put); a projective energy measurement must report the 1s eigenstate.
    if (app.arguments().contains(QStringLiteral("--selftest-energy"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            viewport->toggle_decay();  // OFF: keep the relaxed state stationary
            viewport->set_relaxing();  // cool to 1s
            // 20 s: the n = 5 shell seed is ~orthogonal to 1s, so the descent
            // rides the 1% ground seed (see set_relaxing) -- slower than the
            // old free packet cooled.
            QTimer::singleShot(20000, viewport, [viewport, &app] {
                viewport->measure_energy_now();
                QTimer::singleShot(1500, viewport, [viewport, &app] {
                    const int idx = viewport->last_measured_index();
                    const bool pass = idx == kS1;
                    std::fprintf(
                        stderr, "selftest-energy: measured %s  [%s]\n",
                        idx >= 0 ? kStateSpec[static_cast<std::size_t>(idx)].name
                                 : "outside-manifold",
                        pass ? "PASS" : "FAIL");
                    app.exit(pass ? 0 : 1);
                });
            });
        });
    }

    // Static-E arc: relax to 1s (<z> ~ 0), switch on a sub-ionization +z
    // field, require <z> to shift measurably (Stark polarization).
    if (app.arguments().contains(QStringLiteral("--selftest-efield"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            viewport->toggle_decay();  // OFF: keep the state put
            viewport->set_relaxing();  // cool to 1s
            QTimer::singleShot(20000, viewport, [viewport, &app] {
                const double z0 = viewport->mean_z();
                viewport->set_real_time();
                viewport->set_efield_e0(0.02);  // sub-ionization: clean polarization
                QTimer::singleShot(15000, viewport, [viewport, &app, z0] {
                    const double z1 = viewport->mean_z();
                    const bool pass = std::abs(z1 - z0) > 0.03;
                    std::fprintf(stderr,
                                 "selftest-efield: <z> %.4f -> %.4f Bohr  [%s]\n",
                                 z0, z1, pass ? "PASS" : "FAIL");
                    app.exit(pass ? 0 : 1);
                });
            });
        });
    }

    // Pump arc: relax to 1s, laser ON (Z), require peak P(2pz) >= 0.5; then
    // decay ON and require >= 2 photons (a ground-start run without the
    // pump emits zero, so 2 is unambiguous).
    if (app.arguments().contains(QStringLiteral("--selftest-rabi"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            viewport->toggle_decay();  // OFF: study the clean coherent flop
            viewport->set_relaxing();  // cool to 1s
            QTimer::singleShot(20000, viewport, [viewport, &app] {
                viewport->set_real_time();
                viewport->toggle_laser();  // cached: instant
                // 256^3 runs ~3 au/s of sim time: the half-flop (pi/Omega
                // ~ 79 au) needs most of this window.
                QTimer::singleShot(60000, viewport, [viewport, &app] {
                    const double peak = viewport->peak_excited_population();
                    std::fprintf(stderr, "selftest-rabi: peak P(2pz) = %.3f  [%s]\n",
                                 peak, peak >= 0.5 ? "PASS" : "FAIL");
                    if (peak < 0.5) {
                        app.exit(1);
                        return;
                    }
                    const long long baseline = viewport->photon_count();
                    viewport->toggle_decay();  // back ON: fluorescence
                    QTimer::singleShot(180000, viewport,
                                       [viewport, &app, baseline] {
                        const long long fresh =
                            viewport->photon_count() - baseline;
                        std::fprintf(stderr,
                                     "selftest-rabi: photons = %lld  [%s]\n",
                                     fresh, fresh >= 2 ? "PASS" : "FAIL");
                        app.exit(fresh >= 2 ? 0 : 1);
                    });
                });
            });
        });
    }

    // Cascade arc: excite 3d_z2 and require >= 2 photons -- 3d cannot reach
    // 1s directly (dl = 2), so two photons prove 3d -> 2p -> 1s fired.
    if (app.arguments().contains(QStringLiteral("--selftest-cascade"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            const long long baseline = viewport->photon_count();
            viewport->excite_n3();  // first in the cycle: 3d_z2
            QTimer::singleShot(90000, viewport, [viewport, &app, baseline] {
                const long long fresh = viewport->photon_count() - baseline;
                std::fprintf(stderr, "selftest-cascade: photons = %lld  [%s]\n",
                             fresh, fresh >= 2 ? "PASS" : "FAIL");
                app.exit(fresh >= 2 ? 0 : 1);
            });
        });
    }

    // Magnetic arc: prepare 2p_x (decay off), B along z, require P(2p_y) to
    // rise past 0.3 -- proving psi ITSELF precesses. Probed periodically so
    // the sin^2 oscillation phase cannot alias the verdict.
    if (app.arguments().contains(QStringLiteral("--selftest-magnetic"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            viewport->toggle_decay();            // decay OFF: pure precession
            viewport->debug_prepare_state(kP2X);  // psi = 2p_x
            // Modest field: a large B would diamagnetically deform the state
            // and cap the overlap. Magnetic stepping is slow (~0.4 au/s), so
            // the window is long.
            viewport->set_bfield_b(0.08);         // B along z (default axis)
            auto max_py = std::make_shared<double>(0.0);
            auto* probe = new QTimer(viewport);
            QObject::connect(probe, &QTimer::timeout, viewport, [viewport, max_py] {
                *max_py = std::max(*max_py, viewport->probe_population(kP2Y));
            });
            probe->start(1500);
            QTimer::singleShot(90000, viewport, [&app, max_py, probe] {
                probe->stop();
                std::fprintf(stderr, "selftest-magnetic: max P(2p_y) = %.3f  [%s]\n",
                             *max_py, *max_py > 0.3 ? "PASS" : "FAIL");
                app.exit(*max_py > 0.3 ? 0 : 1);
            });
        });
    }

    // Manifold arc: deterministic channel-table checks (selection rule, 2p
    // degeneracy, ordering, cascade, dl rule), then an X-pol pump from 1s --
    // its photons can only flow through 2p_x, proving non-2p_z channels fire.
    if (app.arguments().contains(QStringLiteral("--selftest-manifold"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            const double a_pz = viewport->channel_a(kP2Z, kS1);
            const double a_px = viewport->channel_a(kP2X, kS1);
            const double a_2s1s = viewport->channel_a(kS2, kS1);
            std::fprintf(stderr,
                         "selftest-manifold: A(2pz->1s)=%.3e A(2px->1s)=%.3e "
                         "A(2s->1s)=%.3e  E(1s)=%.4f E(2pz)=%.4f E(2s)=%.4f\n",
                         a_pz, a_px, a_2s1s, viewport->state_energy(kS1),
                         viewport->state_energy(kP2Z), viewport->state_energy(kS2));
            const bool selection = a_pz > 0.0 && a_2s1s < 1e-3 * a_pz;
            const bool degeneracy = a_pz > 0.0 && std::abs(a_px / a_pz - 1.0) < 0.05;
            const bool ordering =
                viewport->state_energy(kS1) < viewport->state_energy(kP2Z) &&
                viewport->state_energy(kS1) < viewport->state_energy(kS2);
            // The n = 3 shell: cascade paths open, Delta-l selection
            // rules hold (3s -> 1s and 3d -> 1s are E1-forbidden).
            const double a_3s2p = viewport->channel_a(k3S, kP2Z);
            const double a_3d2p = viewport->channel_a(k3DZ0, kP2Z);
            const double a_3s1s = viewport->channel_a(k3S, kS1);
            const double a_3d1s = viewport->channel_a(k3DZ0, kS1);
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
                app.exit(1);
                return;
            }
            viewport->set_relaxing();  // cool to 1s for the X-pol pump
            QTimer::singleShot(20000, viewport, [viewport, &app] {
                viewport->set_real_time();
                viewport->toggle_laser();  // Z (cached: no block)
                viewport->toggle_laser();  // -> X
                const long long baseline = viewport->photon_count();
                // Two X-pol photons need ~170 au of sim time (~2 half-flops
                // + accelerated lifetimes); at ~1.5 au/s a 60 s window was
                // arithmetically unsatisfiable -- 180 s has real margin.
                QTimer::singleShot(180000, viewport, [viewport, &app, baseline] {
                    const long long fresh = viewport->photon_count() - baseline;
                    std::fprintf(stderr,
                                 "selftest-manifold: x-pol photons = %lld  [%s]\n",
                                 fresh, fresh >= 2 ? "PASS" : "FAIL");
                    app.exit(fresh >= 2 ? 0 : 1);
                });
            });
        });
    }

    // Tunneling arc (main forces --scene=tunnel): the packet launched at
    // x = -30 with v = 0.5 reaches the slab at ~60 au; the transmitted lobe
    // is fully past it well before ~150 au (~2.5 min at ~1 au/s). Assert a
    // classically-forbidden transmitted fraction in a sane band.
    if (app.arguments().contains(QStringLiteral("--selftest-tunnel"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            QTimer::singleShot(180000, viewport, [viewport, &app] {
                const double t = viewport->tunnel_transmitted_max();
                const bool pass = t > 1e-3 && t < 0.9;
                std::fprintf(stderr, "selftest-tunnel: max T = %.4f  [%s]\n", t,
                             pass ? "PASS" : "FAIL");
                app.exit(pass ? 0 : 1);
            });
        });
    }
}

}  // namespace ses_shell
