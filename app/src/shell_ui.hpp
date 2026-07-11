#pragma once

// The discoverable-controls toolbar: buttons mirroring the hotkeys plus the
// two field sliders. Buttons keep Qt::NoFocus so the keys stay live on the
// viewport. Templated on the viewport type so main() stays a shell (same
// pattern as selftest_arcs.hpp).

#include <QLabel>
#include <QMainWindow>
#include <QObject>
#include <QSlider>
#include <QString>
#include <QToolBar>

namespace ses_shell {

template <typename ViewportT>
void build_control_bar(QMainWindow& window, ViewportT* viewport) {
    QToolBar* controls = window.addToolBar(QStringLiteral("Controls"));
    controls->setMovable(false);
    controls->addAction(QStringLiteral("Measure (M)"), viewport,
                        [viewport] { viewport->measure_now(); });
    controls->addAction(QStringLiteral("Measure E (E)"), viewport,
                        [viewport] { viewport->measure_energy_now(); });
    controls->addSeparator();
    controls->addAction(QStringLiteral("Real time (1)"), viewport,
                        [viewport] { viewport->set_real_time(); });
    controls->addAction(QStringLiteral("Relax to ground state (2)"), viewport,
                        [viewport] { viewport->set_relaxing(); });
    controls->addAction(QStringLiteral("Relax to 2p (3)"), viewport,
                        [viewport] { viewport->relax_to_excited(); });
    controls->addAction(QStringLiteral("Relax to 2s (4)"), viewport,
                        [viewport] { viewport->relax_to_2s(); });
    controls->addAction(QStringLiteral("Excite n=3/4 (5)"), viewport,
                        [viewport] { viewport->excite_n3(); });
    controls->addAction(QStringLiteral("Decay (D)"), viewport,
                        [viewport] { viewport->toggle_decay(); });
    controls->addAction(QStringLiteral("Laser (L)"), viewport,
                        [viewport] { viewport->toggle_laser(); });
    // Static E-field (+z) slider: 0 = off, full-scale = 0.1 au. A live label
    // shows the value; the tooltip carries the physics.
    controls->addWidget(new QLabel(QStringLiteral(" E-field +z ")));
    {
        constexpr double kMaxEfield = 0.1;  // au at full slider
        auto* efield_val = new QLabel(QStringLiteral("off      "));
        efield_val->setMinimumWidth(96);
        auto* efield_slider = new QSlider(Qt::Horizontal);
        efield_slider->setRange(0, 100);
        efield_slider->setFixedWidth(140);
        efield_slider->setFocusPolicy(Qt::NoFocus);  // keep the hotkeys live
        efield_slider->setToolTip(QStringLiteral(
            "Static uniform E-field along +z (Stark). 0 = off; full = 0.1 au "
            "(~5.1e10 V/m).\nThe 1s ground state barely moves below ~0.03 au, "
            "then field-ionizes."));
        QObject::connect(
            efield_slider, &QSlider::valueChanged, viewport,
            [viewport, efield_val](int val) {
                const double e0 = val / 100.0 * kMaxEfield;
                viewport->set_efield_e0(e0);
                efield_val->setText(
                    e0 > 0.0 ? QStringLiteral("%1 au / %2 V/m")
                                   .arg(e0, 0, 'f', 3)
                                   .arg(e0 * 5.14220674e11, 0, 'e', 1)
                             : QStringLiteral("off"));
            });
        controls->addWidget(efield_slider);
        controls->addWidget(efield_val);
    }
    // Magnetic field: axis cycle (z -> x -> y) + strength slider; psi
    // evolves under the proper minimal-coupling Hamiltonian.
    {
        constexpr double kMaxB = 0.2;  // au at full slider
        auto axis_text = [](int a) {
            return a == 2 ? QStringLiteral(" B z ")
                          : (a == 0 ? QStringLiteral(" B x ") : QStringLiteral(" B y "));
        };
        QLabel* b_axis_label = new QLabel(axis_text(2));
        controls->addWidget(b_axis_label);
        controls->addAction(QStringLiteral("axis"), viewport,
                            [viewport, b_axis_label, axis_text] {
                                viewport->toggle_bfield_axis();
                                b_axis_label->setText(axis_text(viewport->bfield_axis()));
                            });
        auto* b_val = new QLabel(QStringLiteral("off"));
        b_val->setMinimumWidth(60);
        auto* b_slider = new QSlider(Qt::Horizontal);
        b_slider->setRange(0, 100);
        b_slider->setFixedWidth(140);
        b_slider->setFocusPolicy(Qt::NoFocus);
        b_slider->setToolTip(QStringLiteral(
            "Magnetic field along the chosen axis (z or x). The cloud precesses "
            "(Larmor) at omega = B/2. Prepare a p_x / d_xy state to see it rotate; "
            "s and p_z do not precess about z."));
        QObject::connect(b_slider, &QSlider::valueChanged, viewport,
                         [viewport, b_val](int val) {
                             const double b = val / 100.0 * kMaxB;
                             viewport->set_bfield_b(b);
                             b_val->setText(b > 0.0 ? QStringLiteral("%1 au").arg(b, 0, 'f', 3)
                                                    : QStringLiteral("off"));
                         });
        controls->addWidget(b_slider);
        controls->addWidget(b_val);
    }
    controls->addSeparator();
    controls->addAction(QStringLiteral("Reset packet (R)"), viewport,
                        [viewport] { viewport->reset_simulation(); });
    controls->addAction(QStringLiteral("Cloud/Surface (Tab)"), viewport,
                        [viewport] { viewport->toggle_view_mode(); });
    controls->addAction(QStringLiteral("Pause (Space)"), viewport,
                        [viewport] { viewport->toggle_pause(); });
}

}  // namespace ses_shell
