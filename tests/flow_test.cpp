// Probability current / Bohmian velocity + streakline fade (ses.flow).
// The current's handedness is the physical anchor: e^{+i phi} (m=+1) must
// circulate counterclockwise about z -- the same convention the L_z
// measurement labels +m (measurement.hpp).

#include <core/complex.hpp>
#include <core/vec.hpp>

#include <gtest/gtest.h>

import ses.flow;

namespace {

using ses::Complex;
using ses::Vec3d;

// Plane wave psi = e^{i k x}: d_x psi = i k psi, so
// j = Im(conj(psi) i k psi) = k |psi|^2. At x = 0 (psi = 1): j_x = k.
TEST(Flow, PlaneWaveCurrentIsKTimesDensity) {
    const double k = 2.0;
    const Complex<double> psi{1.0, 0.0};
    const Complex<double> dx{0.0, k};  // i k
    const Vec3d j = ses::probability_current(psi, dx, {0.0, 0.0}, {0.0, 0.0});
    EXPECT_NEAR(j.x, k, 1e-12);
    EXPECT_NEAR(j.y, 0.0, 1e-12);
    EXPECT_NEAR(j.z, 0.0, 1e-12);
}

// m = +1 ring psi ~ x + i y (~ e^{i phi}): at (x=1, y=0) the current must
// point +y -- counterclockwise about +z, the right-hand sense of L_z = +hbar.
// grad(x + i y) = (1, i, 0).
TEST(Flow, RingStateM1CirculatesCounterclockwise) {
    const Complex<double> psi{1.0, 0.0};  // value at (1, 0)
    const Complex<double> dx{1.0, 0.0};
    const Complex<double> dy{0.0, 1.0};
    const Vec3d j = ses::probability_current(psi, dx, dy, {0.0, 0.0});
    EXPECT_NEAR(j.x, 0.0, 1e-12);
    EXPECT_GT(j.y, 0.0);  // +y at +x == counterclockwise == L_z > 0
}

// The conjugate m = -1 ring (x - i y) must circulate the OTHER way (-y at +x).
TEST(Flow, RingStateMinus1CirculatesClockwise) {
    const Complex<double> psi{1.0, 0.0};
    const Complex<double> dx{1.0, 0.0};
    const Complex<double> dy{0.0, -1.0};  // grad(x - i y) = (1, -i, 0)
    const Vec3d j = ses::probability_current(psi, dx, dy, {0.0, 0.0});
    EXPECT_LT(j.y, 0.0);  // -y at +x == clockwise == L_z < 0
}

// A real (standing-wave) state carries no current, regardless of its gradient.
TEST(Flow, RealStateHasNoCurrent) {
    const Complex<double> psi{0.7, 0.0};
    const Complex<double> dx{-0.3, 0.0};
    const Complex<double> dy{0.5, 0.0};
    const Complex<double> dz{0.1, 0.0};
    const Vec3d j = ses::probability_current(psi, dx, dy, dz);
    EXPECT_NEAR(j.x, 0.0, 1e-15);
    EXPECT_NEAR(j.y, 0.0, 1e-15);
    EXPECT_NEAR(j.z, 0.0, 1e-15);
}

// v = j / rho; at a node (rho -> 0) it must stay finite (guarded), not blow up.
TEST(Flow, BohmianVelocityGuardsNodes) {
    const Complex<double> zero{0.0, 0.0};
    const Complex<double> dx{5.0, 5.0};
    const Vec3d v = ses::bohmian_velocity(zero, dx, {0.0, 0.0}, {0.0, 0.0});
    EXPECT_EQ(v.x, 0.0);
    EXPECT_EQ(v.y, 0.0);
    EXPECT_EQ(v.z, 0.0);
}

// Trail fade: tail transparent, head opaque, monotone in between.
TEST(Flow, TrailFadeMonotoneTailToHead) {
    EXPECT_NEAR(ses::trail_fade(0, 40), 0.0, 1e-12);
    EXPECT_NEAR(ses::trail_fade(39, 40), 1.0, 1e-12);
    EXPECT_LT(ses::trail_fade(10, 40), ses::trail_fade(20, 40));
    EXPECT_DOUBLE_EQ(ses::trail_fade(0, 1), 1.0);  // degenerate length
}

}  // namespace
