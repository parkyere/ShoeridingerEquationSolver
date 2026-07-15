// RED: the volume-rendering math that the ray-marching shader will mirror.
// The GLSL cannot be unit-tested, so every formula it uses is specified and
// verified HERE first; the shader is then a line-by-line transcription.
//
//  - ray_box: slab intersection, returns the raw [t_near, t_far] interval
//    (t_near < 0 when the ray starts inside; the caller clamps).
//  - sample_alpha: Beer-Lambert emission-absorption opacity per step,
//    alpha = 1 - exp(-absorbance * density * step).
//  - composite_front_to_back: premultiplied front-to-back accumulation
//    C += (1-A) a c,  A += (1-A) a.
//  - phase_lut: the cyclic phase colormap baked into a lookup table so the
//    GPU samples the TESTED colormap instead of a re-derived formula.

#include <core/vec.hpp>
#include <core/volume.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <vector>

import ses.colormap;

namespace {

using ses::Rgb;
using ses::Vec3d;

const Vec3d kBoxMin{-1.0, -1.0, -1.0};
const Vec3d kBoxMax{1.0, 1.0, 1.0};

TEST(RayBox, HitsHeadOnFromOutside) {
    const ses::RayHit h = ses::ray_box(Vec3d{-3.0, 0.0, 0.0}, Vec3d{1.0, 0.0, 0.0},
                                       kBoxMin, kBoxMax);
    ASSERT_TRUE(h.hit);
    EXPECT_DOUBLE_EQ(h.t_near, 2.0);
    EXPECT_DOUBLE_EQ(h.t_far, 4.0);
}

TEST(RayBox, MissesBesideTheBox) {
    // Travels parallel to x at y = 2: outside the y-slab forever.
    const ses::RayHit h = ses::ray_box(Vec3d{-3.0, 2.0, 0.0}, Vec3d{1.0, 0.0, 0.0},
                                       kBoxMin, kBoxMax);
    EXPECT_FALSE(h.hit);
}

TEST(RayBox, MissesPastTheCorner) {
    // Diagonal in the xy-plane that leaves the y-slab before entering the
    // x-slab: intervals [2*sqrt(2), 4*sqrt(2)] vs [-sqrt(2), sqrt(2)] are
    // disjoint.
    const double s = 1.0 / std::sqrt(2.0);
    const ses::RayHit h = ses::ray_box(Vec3d{-3.0, 0.0, 0.0}, Vec3d{s, s, 0.0},
                                       kBoxMin, kBoxMax);
    EXPECT_FALSE(h.hit);
}

TEST(RayBox, StartingInsideGivesNegativeEntry) {
    const ses::RayHit h = ses::ray_box(Vec3d{0.0, 0.0, 0.0}, Vec3d{0.0, 0.0, 1.0},
                                       kBoxMin, kBoxMax);
    ASSERT_TRUE(h.hit);
    EXPECT_DOUBLE_EQ(h.t_near, -1.0);
    EXPECT_DOUBLE_EQ(h.t_far, 1.0);
}

TEST(RayBox, DiagonalThroughCorners) {
    const double inv = 1.0 / std::sqrt(3.0);
    const ses::RayHit h = ses::ray_box(Vec3d{-2.0, -2.0, -2.0}, Vec3d{inv, inv, inv},
                                       kBoxMin, kBoxMax);
    ASSERT_TRUE(h.hit);
    EXPECT_NEAR(h.t_near, std::sqrt(3.0), 1e-12);
    EXPECT_NEAR(h.t_far, 3.0 * std::sqrt(3.0), 1e-12);
}

TEST(RaySphere, HitsHeadOn) {
    const ses::RayHit h =
        ses::ray_sphere(Vec3d{-3.0, 0.0, 0.0}, Vec3d{1.0, 0.0, 0.0}, Vec3d{}, 1.0);
    ASSERT_TRUE(h.hit);
    EXPECT_DOUBLE_EQ(h.t_near, 2.0);
    EXPECT_DOUBLE_EQ(h.t_far, 4.0);
}

TEST(RaySphere, MissesBeside) {
    const ses::RayHit h =
        ses::ray_sphere(Vec3d{-3.0, 2.0, 0.0}, Vec3d{1.0, 0.0, 0.0}, Vec3d{}, 1.0);
    EXPECT_FALSE(h.hit);
}

TEST(RaySphere, TangentTouchesAtOnePoint) {
    // Grazing ray at y = 1 exactly: discriminant 0, t_near == t_far.
    const ses::RayHit h =
        ses::ray_sphere(Vec3d{-3.0, 1.0, 0.0}, Vec3d{1.0, 0.0, 0.0}, Vec3d{}, 1.0);
    ASSERT_TRUE(h.hit);
    EXPECT_DOUBLE_EQ(h.t_near, 3.0);
    EXPECT_DOUBLE_EQ(h.t_far, 3.0);
}

TEST(RaySphere, StartingInsideGivesNegativeEntry) {
    const ses::RayHit h =
        ses::ray_sphere(Vec3d{}, Vec3d{0.0, 0.0, 1.0}, Vec3d{}, 1.0);
    ASSERT_TRUE(h.hit);
    EXPECT_DOUBLE_EQ(h.t_near, -1.0);
    EXPECT_DOUBLE_EQ(h.t_far, 1.0);
}

TEST(RaySphere, OffsetCenter) {
    const ses::RayHit h = ses::ray_sphere(Vec3d{0.0, 0.0, 0.0}, Vec3d{1.0, 0.0, 0.0},
                                          Vec3d{5.0, 0.0, 0.0}, 2.0);
    ASSERT_TRUE(h.hit);
    EXPECT_DOUBLE_EQ(h.t_near, 3.0);
    EXPECT_DOUBLE_EQ(h.t_far, 7.0);
}

TEST(SampleAlpha, ZeroDensityIsTransparent) {
    EXPECT_EQ(ses::sample_alpha(0.0, 5.0, 0.1), 0.0);
}

TEST(SampleAlpha, BeerLambertKnownValue) {
    // 1 - exp(-ln 2) = 1/2.
    EXPECT_NEAR(ses::sample_alpha(1.0, std::numbers::ln2, 1.0), 0.5, 1e-12);
}

TEST(SampleAlpha, MonotoneInDensity) {
    double prev = -1.0;
    for (int i = 0; i <= 10; ++i) {
        const double a = ses::sample_alpha(i / 10.0, 2.0, 0.5);
        EXPECT_GT(a, prev);
        prev = a;
    }
    EXPECT_LT(prev, 1.0);  // never fully saturates
}

TEST(CompositeFrontToBack, EmptyIsTransparent) {
    const ses::Rgba out = ses::composite_front_to_back({});
    EXPECT_EQ(out.r, 0.0);
    EXPECT_EQ(out.g, 0.0);
    EXPECT_EQ(out.b, 0.0);
    EXPECT_EQ(out.a, 0.0);
}

TEST(CompositeFrontToBack, OpaqueFrontOccludesBack) {
    const std::vector<ses::VolumeSample> samples = {
        {Rgb{1.0, 0.0, 0.0}, 1.0},  // opaque red in front
        {Rgb{0.0, 1.0, 0.0}, 1.0},  // green behind: must not contribute
    };
    const ses::Rgba out = ses::composite_front_to_back(samples);
    EXPECT_DOUBLE_EQ(out.r, 1.0);
    EXPECT_DOUBLE_EQ(out.g, 0.0);
    EXPECT_DOUBLE_EQ(out.a, 1.0);
}

TEST(CompositeFrontToBack, PartialBlendMatchesAnalytic) {
    // front (red, 0.5), back (green, 0.5):
    //   C = 0.5 red + (1-0.5) 0.5 green = (0.5, 0.25, 0), A = 0.75.
    const std::vector<ses::VolumeSample> samples = {
        {Rgb{1.0, 0.0, 0.0}, 0.5},
        {Rgb{0.0, 1.0, 0.0}, 0.5},
    };
    const ses::Rgba out = ses::composite_front_to_back(samples);
    EXPECT_DOUBLE_EQ(out.r, 0.5);
    EXPECT_DOUBLE_EQ(out.g, 0.25);
    EXPECT_DOUBLE_EQ(out.b, 0.0);
    EXPECT_DOUBLE_EQ(out.a, 0.75);
}

TEST(PhaseLut, EntriesAreBinCenterPhaseColors) {
    const int n = 256;
    const std::vector<Rgb> lut = ses::phase_lut(n);
    ASSERT_EQ(lut.size(), static_cast<std::size_t>(n));
    for (int i = 0; i < n; i += 17) {  // spot-check across the table
        const double theta =
            -std::numbers::pi + 2.0 * std::numbers::pi * (i + 0.5) / n;
        const Rgb expected = ses::phase_color(theta);
        EXPECT_DOUBLE_EQ(lut[static_cast<std::size_t>(i)].r, expected.r);
        EXPECT_DOUBLE_EQ(lut[static_cast<std::size_t>(i)].g, expected.g);
        EXPECT_DOUBLE_EQ(lut[static_cast<std::size_t>(i)].b, expected.b);
    }
}

}  // namespace
