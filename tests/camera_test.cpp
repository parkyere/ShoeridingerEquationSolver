// RED: hand-rolled 3D math for the renderer -- Vec3d operations and a
// column-major Mat4 with perspective / look_at builders.
//
// Conventions pinned here (GL-style clip conventions, and classic bug
// territory; the Vulkan renderer applies its own y-flip/depth-remap
// clip correction on top of these):
//  - Mat4 storage is COLUMN-MAJOR: element(row r, col c) = m[c*4 + r], so the
//    translation vector of a transform sits at m[12], m[13], m[14].
//  - View space is right-handed, camera looks down -Z.
//  - NDC depth range is [-1, +1] (near plane -> -1, far plane -> +1).

#include <core/vec.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

import ses.camera;

namespace {

using ses::Mat4;
using ses::Vec3d;

TEST(Vec3, DotCrossNormalize) {
    const Vec3d x{1.0, 0.0, 0.0};
    const Vec3d y{0.0, 1.0, 0.0};
    EXPECT_EQ(dot(x, y), 0.0);
    EXPECT_EQ(dot(x, x), 1.0);
    const Vec3d z = cross(x, y);  // right-handed: x cross y = +z
    EXPECT_EQ(z.x, 0.0);
    EXPECT_EQ(z.y, 0.0);
    EXPECT_EQ(z.z, 1.0);
    const Vec3d n = normalized(Vec3d{3.0, 0.0, 4.0});
    EXPECT_DOUBLE_EQ(n.x, 0.6);
    EXPECT_DOUBLE_EQ(n.z, 0.8);
}

TEST(Mat4, IdentityLeavesPointsUnchanged) {
    const Vec3d p = transform_point(Mat4::identity(), Vec3d{1.5, -2.0, 3.0});
    EXPECT_DOUBLE_EQ(p.x, 1.5);
    EXPECT_DOUBLE_EQ(p.y, -2.0);
    EXPECT_DOUBLE_EQ(p.z, 3.0);
}

TEST(Mat4, TranslationStorageIsColumnMajor) {
    const Mat4 t = Mat4::translation(Vec3d{7.0, 8.0, 9.0});
    EXPECT_EQ(t.m[12], 7.0);
    EXPECT_EQ(t.m[13], 8.0);
    EXPECT_EQ(t.m[14], 9.0);
    EXPECT_EQ(t.m[15], 1.0);
    EXPECT_EQ(t.m[3], 0.0);  // NOT row-major: bottom row of column 0 is 0
}

TEST(Mat4, MultiplyComposesRightToLeft) {
    // (translate by +x) * (scale by 2): point (1,1,1) -> scaled (2,2,2) ->
    // translated (5,2,2).
    const Mat4 t = Mat4::translation(Vec3d{3.0, 0.0, 0.0});
    const Mat4 s = Mat4::scale(2.0);
    const Vec3d p = transform_point(t * s, Vec3d{1.0, 1.0, 1.0});
    EXPECT_DOUBLE_EQ(p.x, 5.0);
    EXPECT_DOUBLE_EQ(p.y, 2.0);
    EXPECT_DOUBLE_EQ(p.z, 2.0);
}

TEST(LookAt, MapsWorldIntoViewSpace) {
    // Eye at (0,0,5) looking at the origin, +Y up: forward is -Z.
    const Mat4 view = ses::look_at(Vec3d{0.0, 0.0, 5.0}, Vec3d{}, Vec3d{0.0, 1.0, 0.0});

    const Vec3d eye = transform_point(view, Vec3d{0.0, 0.0, 5.0});
    EXPECT_NEAR(eye.x, 0.0, 1e-12);
    EXPECT_NEAR(eye.y, 0.0, 1e-12);
    EXPECT_NEAR(eye.z, 0.0, 1e-12);

    const Vec3d target = transform_point(view, Vec3d{});
    EXPECT_NEAR(target.z, -5.0, 1e-12);  // straight ahead, down -Z

    const Vec3d right = transform_point(view, Vec3d{1.0, 0.0, 5.0});
    EXPECT_NEAR(right.x, 1.0, 1e-12);  // world +X stays view +X here
    EXPECT_NEAR(right.z, 0.0, 1e-12);

    const Vec3d up = transform_point(view, Vec3d{0.0, 1.0, 5.0});
    EXPECT_NEAR(up.y, 1.0, 1e-12);
}

TEST(Perspective, MapsFrustumToNdc) {
    // fovy = 90 deg, aspect 1, near 1, far 3: cot(fovy/2) = 1.
    const double fovy = std::numbers::pi / 2.0;
    const Mat4 proj = ses::perspective(fovy, 1.0, 1.0, 3.0);

    const Vec3d near_center = transform_point(proj, Vec3d{0.0, 0.0, -1.0});
    EXPECT_NEAR(near_center.z, -1.0, 1e-12);  // near plane -> NDC -1

    const Vec3d far_center = transform_point(proj, Vec3d{0.0, 0.0, -3.0});
    EXPECT_NEAR(far_center.z, 1.0, 1e-12);  // far plane -> NDC +1

    const Vec3d near_edge = transform_point(proj, Vec3d{1.0, 0.0, -1.0});
    EXPECT_NEAR(near_edge.x, 1.0, 1e-12);  // 45-degree half-angle -> NDC edge

    const Vec3d far_edge = transform_point(proj, Vec3d{0.0, -3.0, -3.0});
    EXPECT_NEAR(far_edge.y, -1.0, 1e-12);  // scales with depth
}

TEST(OrbitCamera, EyeOnSphereAroundTarget) {
    // azimuth 0, elevation 0, distance 5 about the origin -> on the +Z axis.
    const Vec3d eye0 = ses::orbit_eye(0.0, 0.0, 5.0, Vec3d{});
    EXPECT_NEAR(eye0.x, 0.0, 1e-12);
    EXPECT_NEAR(eye0.y, 0.0, 1e-12);
    EXPECT_NEAR(eye0.z, 5.0, 1e-12);

    // azimuth 90 deg swings to +X; elevation 90 deg rises to +Y.
    const double half_pi = std::numbers::pi / 2.0;
    const Vec3d eye_x = ses::orbit_eye(half_pi, 0.0, 5.0, Vec3d{});
    EXPECT_NEAR(eye_x.x, 5.0, 1e-12);
    EXPECT_NEAR(eye_x.z, 0.0, 1e-12);

    const Vec3d eye_y = ses::orbit_eye(0.0, half_pi, 5.0, Vec3d{});
    EXPECT_NEAR(eye_y.y, 5.0, 1e-12);
    EXPECT_NEAR(eye_y.z, 0.0, 1e-12);

    // Off the origin: sphere is centered on the target.
    const Vec3d eye_t = ses::orbit_eye(0.0, 0.0, 2.0, Vec3d{1.0, 2.0, 3.0});
    EXPECT_NEAR(eye_t.x, 1.0, 1e-12);
    EXPECT_NEAR(eye_t.y, 2.0, 1e-12);
    EXPECT_NEAR(eye_t.z, 5.0, 1e-12);
}

}  // namespace
