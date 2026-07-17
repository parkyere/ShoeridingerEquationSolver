// RED: colormaps for wavefunction visualization.
//
//  - phase_color(theta): CYCLIC map for arg(psi) in [-pi, pi]. The load-
//    bearing property is periodicity -- color(-pi) == color(+pi) -- so the
//    phase shows no artificial seam (a non-cyclic map like viridis would
//    manufacture a discontinuity there).
//  - magnitude_color(t): sequential dark -> bright map for |psi|^2 in [0,1],
//    monotone in brightness so density reads correctly.


#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

import ses.colormap;

namespace {

using ses::Rgb;

constexpr double kPi = std::numbers::pi;

TEST(PhaseColormap, IsCyclicAtPlusMinusPi) {
    const Rgb a = ses::phase_color(-kPi);
    const Rgb b = ses::phase_color(kPi);
    EXPECT_NEAR(a.r, b.r, 1e-9);
    EXPECT_NEAR(a.g, b.g, 1e-9);
    EXPECT_NEAR(a.b, b.b, 1e-9);
}

TEST(PhaseColormap, ComponentsStayInRange) {
    for (int i = -300; i <= 300; ++i) {
        const Rgb c = ses::phase_color(kPi * i / 300.0);
        EXPECT_GE(c.r, 0.0);
        EXPECT_LE(c.r, 1.0);
        EXPECT_GE(c.g, 0.0);
        EXPECT_LE(c.g, 1.0);
        EXPECT_GE(c.b, 0.0);
        EXPECT_LE(c.b, 1.0);
    }
}

TEST(PhaseColormap, DistinguishesOppositePhases) {
    const Rgb a = ses::phase_color(0.0);
    const Rgb b = ses::phase_color(kPi);
    const double diff =
        std::abs(a.r - b.r) + std::abs(a.g - b.g) + std::abs(a.b - b.b);
    EXPECT_GT(diff, 0.5);
}

TEST(PhaseColormap, IsContinuousIncludingWrap) {
    // Small phase steps produce small color steps everywhere, wrap included.
    const double dtheta = 1e-3;
    for (int i = -300; i <= 300; ++i) {
        const double th = kPi * i / 300.0;
        const Rgb a = ses::phase_color(th);
        const Rgb b = ses::phase_color(th + dtheta);
        const double diff =
            std::abs(a.r - b.r) + std::abs(a.g - b.g) + std::abs(a.b - b.b);
        EXPECT_LT(diff, 0.01) << "jump at theta = " << th;
    }
}

TEST(MagnitudeColormap, EndpointsAreDarkToBright) {
    const Rgb lo = ses::magnitude_color(0.0);
    const Rgb hi = ses::magnitude_color(1.0);
    EXPECT_LT(lo.r + lo.g + lo.b, 0.5);
    EXPECT_GT(hi.r + hi.g + hi.b, 2.5);
}

TEST(MagnitudeColormap, BrightnessIsMonotone) {
    double prev = -1.0;
    for (int i = 0; i <= 100; ++i) {
        const Rgb c = ses::magnitude_color(i / 100.0);
        const double sum = c.r + c.g + c.b;
        EXPECT_GE(sum, prev);
        prev = sum;
    }
}

TEST(MagnitudeColormap, ClampsOutOfRangeInput) {
    const Rgb below = ses::magnitude_color(-0.5);
    const Rgb at0 = ses::magnitude_color(0.0);
    EXPECT_EQ(below.r, at0.r);
    EXPECT_EQ(below.b, at0.b);
    const Rgb above = ses::magnitude_color(1.5);
    const Rgb at1 = ses::magnitude_color(1.0);
    EXPECT_EQ(above.g, at1.g);
}

}  // namespace
