// RED: the pure logic behind the cross-section display feature -- the clip
// plane's ray-interval math, the slice quad geometry, and the slice colour
// mapping. These are the SINGLE SOURCE OF TRUTH the GLSL (volume.frag,
// slice.vert, slice.frag) mirrors textually. Verification posture matches
// volume.frag / ses.volume / volume_test.cpp: the CPU logic is pinned
// here; the fragment shader's transcription is verified by the dump-frame
// composite, NOT by a per-fragment GPU oracle (there is none).

#include <core/complex.hpp>
#include <core/grid.hpp>
#include <core/vec.hpp>

#include <gtest/gtest.h>

#include <cmath>

import ses.colormap;
import ses.cross_section;

namespace {

using ses::Complex;
using ses::Grid1D;
using ses::Grid3D;
using ses::Vec3d;

TEST(ClipRayInterval, ClampsToTheVisibleHalfCrossingForward) {
    // Ray from -5 along +axis; keep sign*(p-offset) <= 0, sign=+1, offset=0
    // => p <= 0 => t <= 5. t_stop clamps 10 -> 5.
    const ses::ClipInterval c = ses::clip_ray_interval(0.0, 10.0, -5.0, 1.0,
                                                       1.0, 0.0);
    EXPECT_TRUE(c.visible);
    EXPECT_NEAR(c.tn, 0.0, 1e-12);
    EXPECT_NEAR(c.t_stop, 5.0, 1e-12);
}

TEST(ClipRayInterval, SignFlipKeepsTheOtherHalf) {
    const ses::ClipInterval c = ses::clip_ray_interval(0.0, 10.0, -5.0, 1.0,
                                                       -1.0, 0.0);
    EXPECT_TRUE(c.visible);
    EXPECT_NEAR(c.tn, 5.0, 1e-12);
    EXPECT_NEAR(c.t_stop, 10.0, 1e-12);
}

TEST(ClipRayInterval, OffsetShiftsThePlane) {
    const ses::ClipInterval c = ses::clip_ray_interval(0.0, 10.0, -5.0, 1.0,
                                                       1.0, 2.0);
    EXPECT_NEAR(c.t_stop, 7.0, 1e-12);  // p <= 2 => t <= 7
}

TEST(ClipRayInterval, ParallelRayOnVisibleSideIsUnchanged) {
    const ses::ClipInterval c = ses::clip_ray_interval(1.0, 9.0, -3.0, 0.0,
                                                       1.0, 0.0);
    EXPECT_TRUE(c.visible);
    EXPECT_NEAR(c.tn, 1.0, 1e-12);
    EXPECT_NEAR(c.t_stop, 9.0, 1e-12);
}

TEST(ClipRayInterval, ParallelRayOnCutSideIsInvisible) {
    const ses::ClipInterval c = ses::clip_ray_interval(1.0, 9.0, 3.0, 0.0,
                                                       1.0, 0.0);
    EXPECT_FALSE(c.visible);
}

TEST(SliceQuadCorner, AllSixCornersLieOnThePlaneAndSpanTheBox) {
    const Grid1D ax{-8.0, 8.0, 16};
    const Grid3D g{ax, ax, ax};
    const int axis = 2;      // z-normal
    const double offset = 3.0;
    double umin = 1e9, umax = -1e9, wmin = 1e9, wmax = -1e9;
    for (int k = 0; k < 6; ++k) {
        const Vec3d p = ses::slice_quad_corner(axis, offset, g, k);
        EXPECT_NEAR(p.z, offset, 1e-12);  // on the z = offset plane
        umin = std::min(umin, p.x);
        umax = std::max(umax, p.x);
        wmin = std::min(wmin, p.y);
        wmax = std::max(wmax, p.y);
    }
    EXPECT_NEAR(umin, -8.0, 1e-12);
    EXPECT_NEAR(umax, 8.0, 1e-12);
    EXPECT_NEAR(wmin, -8.0, 1e-12);
    EXPECT_NEAR(wmax, 8.0, 1e-12);
}

TEST(SliceShade, DensityModeIsMagnitudeColorAndBaseAlpha) {
    // |psi|^2 * inv_peak = 0.5 -> magnitude_color(0.5), alpha 0.45+0.5*0.5.
    const Complex<double> psi{std::sqrt(0.5), 0.0};
    const ses::SliceShade s = ses::slice_shade(0, psi, 1.0);
    const ses::Rgb m = ses::magnitude_color(0.5);
    EXPECT_NEAR(s.col.r, m.r, 1e-12);
    EXPECT_NEAR(s.col.g, m.g, 1e-12);
    EXPECT_NEAR(s.col.b, m.b, 1e-12);
    EXPECT_NEAR(s.alpha, 0.70, 1e-12);
}

TEST(SliceShade, ReModeIsDivergingBySign) {
    const ses::SliceShade pos = ses::slice_shade(1, Complex<double>{0.8, 0.0},
                                                 1.0);
    const ses::SliceShade neg = ses::slice_shade(1, Complex<double>{-0.8, 0.0},
                                                 1.0);
    // Positive lobe warm (r > b), negative lobe cool (b > r).
    EXPECT_GT(pos.col.r, pos.col.b);
    EXPECT_GT(neg.col.b, neg.col.r);
    EXPECT_NEAR(pos.alpha, 0.85, 1e-12);  // 0.45 + 0.5*0.8
    // Zero amplitude collapses to the dark midpoint.
    const ses::SliceShade zero = ses::slice_shade(1, Complex<double>{0.0, 0.0},
                                                  1.0);
    EXPECT_NEAR(zero.col.r, 0.03, 1e-12);
    EXPECT_NEAR(zero.col.b, 0.03, 1e-12);
}

TEST(SliceShade, PhaseModeTintsThePhaseWheelByMagnitude) {
    // psi = i*mag -> arg = +pi/2; brightness = sqrt(dens).
    const double mag = std::sqrt(0.25);  // dens = 0.25, bright = 0.5
    const Complex<double> psi{0.0, mag};
    const ses::SliceShade s = ses::slice_shade(2, psi, 1.0);
    const ses::Rgb wheel = ses::phase_color(std::atan2(mag, 0.0));
    const double bright = 0.5;
    const double tint = 0.25 + 0.75 * bright;
    EXPECT_NEAR(s.col.r, wheel.r * tint, 1e-12);
    EXPECT_NEAR(s.col.g, wheel.g * tint, 1e-12);
    EXPECT_NEAR(s.col.b, wheel.b * tint, 1e-12);
    EXPECT_NEAR(s.alpha, 0.45 + 0.5 * bright, 1e-12);
}

}  // namespace
