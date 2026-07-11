#pragma once

// Verification + selftest arcs: every --dump-frame* and --selftest-*
// command-line arc, registered against the live viewport. Templated on the
// viewport type so the shell class can stay
// private to main.cpp; the arcs consume only its public control/probe API.
// Each arc waits for the startup atlas build (run_when_manifold_ready) and
// then CHAINS timers, so a slower GPU stretches the run instead of
// false-failing a wall-clock verdict.

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
    // Render verification hook: once the manifold is up (the cloud shows the
    // resumed wavepacket), grab the composited frame to frame_dump.bmp and
    // exit. grabFramebuffer renders offscreen + reads the color buffer back,
    // so this verifies the whole render path (ses_vk scene + the Qt blit)
    // end to end. BMP because the lean static Qt is built without the png
    // feature (no libpng).
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
    // Same, from INSIDE the box (distance 4 Bohr in the +-80 box): the volume
    // pass rasterizes the proxy's back faces (front-face culled), which is
    // only exercised by an interior eye -- if the culling/winding choice were
    // wrong under Vulkan, this frame would lose the cloud entirely.
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

    // Headless-ish regression of the decay demo arc: prepare 2p, return to
    // real time, enable decay, and require at least one quantum jump.
    // (After the first jump the atom sits in 1s with P_e ~ 0, so exactly one
    // photon is the physically expected outcome without a re-pump laser.)
    // Selftest arcs wait for the startup atlas build (run_when_manifold_ready)
    // and are then CHAINED, so a slower GPU stretches the run instead of
    // false-failing a wall-clock verdict. Decay is ON by default;
    // photon verdicts count from a baseline captured at the arc's start.
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

    // Headless-ish regression of the energy-measurement feature: relax to 1s
    // (decay OFF so the prepared state stays put), then a projective energy
    // measurement must collapse onto -- and report -- the 1s eigenstate.
    if (app.arguments().contains(QStringLiteral("--selftest-energy"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            viewport->toggle_decay();  // OFF: keep the relaxed state stationary
            viewport->set_relaxing();  // cool to 1s
            QTimer::singleShot(12000, viewport, [viewport, &app] {
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

    // Headless-ish regression of the static E-field: relax to 1s (symmetric,
    // <z> ~ 0), switch on a sub-ionization +z field, and require the cloud to
    // polarize -- <z> shifts measurably off center (Stark). Proves the field
    // actually acts on the cloud.
    if (app.arguments().contains(QStringLiteral("--selftest-efield"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            viewport->toggle_decay();  // OFF: keep the state put
            viewport->set_relaxing();  // cool to 1s
            QTimer::singleShot(12000, viewport, [viewport, &app] {
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

    // Headless-ish regression of the pump demo: relax to 1s, laser ON
    // (Z-pol), require a Rabi peak P(2pz) >= 0.5; then decay ON as well and
    // require >= 2 photons -- repeated absorb/emit cycles. A ground-start
    // run WITHOUT the pump emits zero photons, so 2 is unambiguous.
    if (app.arguments().contains(QStringLiteral("--selftest-rabi"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            viewport->toggle_decay();  // OFF: study the clean coherent flop
            viewport->set_relaxing();  // cool to 1s
            QTimer::singleShot(11500, viewport, [viewport, &app] {
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

    // Cascade regression: excite 3d_z2 instantly (key-5 path) and
    // require at least two photons: 3d cannot reach 1s directly (dl = 2),
    // so two photons prove the chain 3d -> 2p -> 1s fired through the
    // multi-level channel table.
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

    // Magnetic regression: a field along z rotates 2p_x -> 2p_y at
    // omega_L = B/2. Prepare 2p_x (decay off, for pure precession), turn B on,
    // and require P(2p_y) to rise past 0.3 -- proving psi ITSELF precesses (the
    // old display trick left psi pristine, so P(2p_y) would stay 0). Probed
    // periodically so the sin^2 oscillation phase cannot alias the verdict.
    if (app.arguments().contains(QStringLiteral("--selftest-magnetic"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            viewport->toggle_decay();            // decay OFF: pure precession
            viewport->debug_prepare_state(kP2X);  // psi = 2p_x
            // Modest field: omega_L = B/2 precession with only a MILD
            // diamagnetic term, so 2p_x rotates substantially into 2p_y (a
            // large B would diamagnetically deform the state and cap the
            // overlap -- itself why the display trick was wrong). Real-time
            // magnetic stepping is slow (~0.4 au/s), so the window is long.
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

    // Manifold regression: (a) deterministic physics of the computed channel
    // table -- the selection rule A(2s->1s) ~ 0 and the 2p degeneracy --
    // then (b) live wiring of the non-2p_z channels: an X-polarized pump
    // from 1s can only fluoresce through 2p_x, so new photons prove the
    // multi-channel trial fires through channels other than 2p_z -> 1s.
    if (app.arguments().contains(QStringLiteral("--selftest-manifold"))) {
        run_when_manifold_ready(viewport, [viewport, &app] {
            // Deterministic physics of the freshly built channel table.
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
            QTimer::singleShot(12000, viewport, [viewport, &app] {
                viewport->set_real_time();
                viewport->toggle_laser();  // Z (cached: no block)
                viewport->toggle_laser();  // -> X
                const long long baseline = viewport->photon_count();
                // Two X-pol fluorescence photons need ~2 pump half-flops of
                // SIM time: the pump starts from P_e = 0, the half-flop is
                // pi/kRabiTargetOmega ~ 79 au, and the display-accelerated
                // 2p lifetime adds ~8 au per cycle -- ~170 au to be safe. At
                // the measured 256^3 sim rate (~1.5 au/s with VkFFT) that is
                // ~115 s of wall clock, so a 60 s window was arithmetically
                // unsatisfiable on this hardware regardless of correctness.
                // 180 s carries the same physics verdict with real margin.
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
}

}  // namespace ses_shell
