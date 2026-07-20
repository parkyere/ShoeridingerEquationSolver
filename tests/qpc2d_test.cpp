// RED: the quantum-point-contact staircase. Transverse modes open every
// lambda/2 = pi/k0 of gap width: below the first threshold the wall
// insulates (tunneling dribble only); above it a channel conducts; the
// next lambda/2 opens a second channel and the transmitted flux steps UP
// again. Wide coherent front, edge CAPs, right-cap flux = transmission.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <vector>

import ses.scenario.qpc2d_director;
import ses.field;
import ses.grid;
import ses.propagator;

namespace {

TEST(Qpc2d, TransmissionClimbsTheModeStaircase) {
    const ses::Grid3D g{ses::Grid1D{-30.0, 30.0, 256},
                        ses::Grid1D{-20.0, 20.0, 128},
                        ses::Grid1D{0.0, 2.0, 1}};
    const double k0 = 1.0;  // lambda/2 = pi: thresholds ~3.1, ~6.3
    const double dt = 0.01;
    const double cap_w = 5.0;
    const double cap_w0 = 4.0;
    std::vector<double> cap(static_cast<std::size_t>(g.size()), 1.0);
    for (int j = 0; j < g.y.n; ++j) {
        const double y = g.y.coord(j);
        const double dy = std::min(y - g.y.xmin, g.y.xmax - y);
        for (int i = 0; i < g.x.n; ++i) {
            const double x = g.x.coord(i);
            const double dx = std::min(x - g.x.xmin, g.x.xmax - x);
            double wsum = 0.0;
            if (dx < cap_w) {
                const double t = 1.0 - dx / cap_w;
                wsum += cap_w0 * t * t;
            }
            if (dy < cap_w) {
                const double t = 1.0 - dy / cap_w;
                wsum += cap_w0 * t * t;
            }
            cap[static_cast<std::size_t>(g.flat(i, j, 0))] =
                std::exp(-wsum * dt);
        }
    }
    auto transmission = [&](double gap) {
        const std::vector<double> v =
            ses_shell::qpc_potential(g, gap, 0.0, 1.5, 40.0, 0.8);
        const ses::SplitOperator3D prop{g, v, dt};
        ses::Field3D psi{g};
        for (int j = 0; j < g.y.n; ++j) {
            const double y = g.y.coord(j);
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i);
                const double dx0 = x + 18.0;
                psi(i, j, 0) =
                    std::exp(-dx0 * dx0 / (4.0 * 4.0 * 4.0) -
                             y * y / (4.0 * 8.0 * 8.0)) *
                    std::complex<double>{std::cos(k0 * x),
                                         std::sin(k0 * x)};
            }
        }
        ses::normalize(psi);
        const double cell = g.x.spacing() * g.y.spacing() * g.z.spacing();
        double transmitted = 0.0;
        for (int s = 0; s < 5000; ++s) {  // t = 50: transit + tail
            prop.step(psi, 1);
            for (int j = 0; j < g.y.n; ++j) {
                for (int i = 0; i < g.x.n; ++i) {
                    const std::size_t c =
                        static_cast<std::size_t>(g.flat(i, j, 0));
                    const double m = cap[c];
                    if (m < 1.0 && g.x.coord(i) > 1.5) {
                        transmitted +=
                            std::norm(psi.data()[c]) * (1.0 - m * m) * cell;
                    }
                    psi.data()[c] *= m;
                }
            }
        }
        return transmitted;
    };
    const double t_closed = transmission(2.0);   // below the first mode
    const double t_one = transmission(4.5);      // one channel open
    const double t_plateau = transmission(5.5);  // still one channel
    const double t_two = transmission(8.0);      // two channels
    std::printf("qpc: T(2.0) %.4f, T(4.5) %.4f, T(5.5) %.4f, T(8.0) %.4f\n",
                t_closed, t_one, t_plateau, t_two);
    EXPECT_LT(t_closed, 0.25 * t_one);            // insulating foot
    EXPECT_GT(t_one, 0.02);                       // the channel conducts
    EXPECT_LT(std::abs(t_plateau - t_one),
              0.35 * t_one);                      // first plateau
    EXPECT_GT(t_two, 1.4 * t_one);                // second channel opens
}

}  // namespace
