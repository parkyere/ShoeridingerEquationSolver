// RED: the quantum-billiard stage. stadium_sdf geometry (half_len = 0 IS
// the circle; caps and flats measure true distance), and the
// integrable-vs-chaotic caustic contract: a tangential packet in the
// CIRCLE conserves |L| so its TIME-AVERAGED density keeps a dark hole
// inside the caustic radius; the STADIUM's flat walls break L and the
// average fills the center.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <vector>

import ses.scenario.billiard2d_director;
import ses.field;
import ses.grid;
import ses.propagator;

namespace {

TEST(Billiard2D, StadiumSdfGeometry) {
    // half_len = 0: exactly the circle SDF.
    EXPECT_NEAR(ses_shell::stadium_sdf(3.0, 4.0, 0.0, 9.0), -4.0, 1e-12);
    EXPECT_NEAR(ses_shell::stadium_sdf(0.0, 0.0, 6.0, 9.0), -9.0, 1e-12);
    // Beyond the right cap: distance from the cap center (6, 0).
    EXPECT_NEAR(ses_shell::stadium_sdf(17.0, 0.0, 6.0, 9.0), 2.0, 1e-12);
    // Above the flat section: distance from the segment line.
    EXPECT_NEAR(ses_shell::stadium_sdf(0.0, 11.0, 6.0, 9.0), 2.0, 1e-12);
    // Mirror symmetry in both axes.
    EXPECT_DOUBLE_EQ(ses_shell::stadium_sdf(5.0, 3.0, 6.0, 9.0),
                     ses_shell::stadium_sdf(-5.0, 3.0, 6.0, 9.0));
    EXPECT_DOUBLE_EQ(ses_shell::stadium_sdf(5.0, 3.0, 6.0, 9.0),
                     ses_shell::stadium_sdf(5.0, -3.0, 6.0, 9.0));
}

TEST(Billiard2D, CircleKeepsTheCausticHoleTheStadiumFillsIt) {
    const ses::Grid3D g{ses::Grid1D{-20.0, 20.0, 128},
                        ses::Grid1D{-20.0, 20.0, 128},
                        ses::Grid1D{0.0, 2.0, 1}};
    const double R = 9.0;
    const double r0 = 4.5;   // launch radius: the caustic of the circle
    const double k0 = 2.0;
    const double v0 = 24.0;  // wall: V0 min(1, (d/w)^2), quadratic onset
    const double w = 2.0;
    auto run = [&](double half_len) {
        std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
        for (int j = 0; j < g.y.n; ++j) {
            const double y = g.y.coord(j);
            for (int i = 0; i < g.x.n; ++i) {
                const double d = ses_shell::stadium_sdf(g.x.coord(i), y,
                                                        half_len, R);
                if (d > 0.0) {
                    const double t = d / w;
                    v[static_cast<std::size_t>(g.flat(i, j, 0))] =
                        v0 * std::min(1.0, t * t);
                }
            }
        }
        const ses::SplitOperator3D prop{g, v, 0.01};
        ses::Field3D psi{g};
        for (int j = 0; j < g.y.n; ++j) {
            const double y = g.y.coord(j) - r0;  // launch at (0, r0)...
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i);
                psi(i, j, 0) =
                    std::exp(-(x * x + y * y) / (4.0 * 1.5 * 1.5)) *
                    std::complex<double>{std::cos(k0 * x),
                                         std::sin(k0 * x)};  // ...moving +x
            }
        }
        ses::normalize(psi);
        std::vector<double> avg(static_cast<std::size_t>(g.size()), 0.0);
        for (int s = 0; s < 6000; ++s) {
            prop.step(psi, 1);
            for (std::size_t c = 0; c < avg.size(); ++c) {
                avg[c] += std::norm(psi.data()[c]);
            }
        }
        double center = 0.0;
        int n_center = 0;
        double interior = 0.0;
        int n_interior = 0;
        for (int j = 0; j < g.y.n; ++j) {
            const double y = g.y.coord(j);
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i);
                const double a =
                    avg[static_cast<std::size_t>(g.flat(i, j, 0))];
                if (std::hypot(x, y) < 1.5) {
                    center += a;
                    ++n_center;
                }
                if (ses_shell::stadium_sdf(x, y, half_len, R) < -2.0) {
                    interior += a;
                    ++n_interior;
                }
            }
        }
        center /= n_center;
        interior /= n_interior;
        return interior > 0.0 ? center / interior : -1.0;
    };
    const double circle = run(0.0);
    const double stadium = run(6.0);
    std::printf("billiard caustic center/interior: circle %.3f stadium %.3f\n",
                circle, stadium);
    // Integrable: the caustic keeps the center DARK relative to the
    // orbit annulus; chaotic: L is broken and the center lights up.
    EXPECT_GT(circle, 0.0);
    EXPECT_LT(circle, 0.30);
    EXPECT_GT(stadium, 2.0 * circle);
}

}  // namespace
